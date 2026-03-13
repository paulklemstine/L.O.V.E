/**
 * bluesky.js - Bluesky AT Protocol client for browser
 */

const BSKY_API = 'https://bsky.social/xrpc';

/**
 * Detect URLs in text and return Bluesky link facets (byte-indexed).
 */
function detectLinkFacets(text) {
  const encoder = new TextEncoder();
  const facets = [];
  const urlRegex = /https?:\/\/[^\s)]+/g;
  let match;
  while ((match = urlRegex.exec(text)) !== null) {
    const beforeBytes = encoder.encode(text.slice(0, match.index)).byteLength;
    const matchBytes = encoder.encode(match[0]).byteLength;
    facets.push({
      index: { byteStart: beforeBytes, byteEnd: beforeBytes + matchBytes },
      features: [{ $type: 'app.bsky.richtext.facet#link', uri: match[0] }]
    });
  }
  return facets;
}

export class BlueskyClient {
  constructor() {
    this.session = null; // { did, handle, accessJwt, refreshJwt }
    this.pdsUrl = BSKY_API; // actual PDS endpoint (discovered from didDoc)
    this.onSessionChange = null;
  }

  get isLoggedIn() {
    return !!this.session?.accessJwt;
  }

  /**
   * Login to Bluesky.
   */
  async login(identifier, password) {
    const res = await this._fetch('com.atproto.server.createSession', {
      method: 'POST',
      body: { identifier, password },
      noAuth: true
    });
    this.session = {
      did: res.did,
      handle: res.handle,
      accessJwt: res.accessJwt,
      refreshJwt: res.refreshJwt
    };
    // Discover actual PDS endpoint from didDoc
    this._extractPdsUrl(res.didDoc);
    this.onSessionChange?.(this.session);
    return this.session;
  }

  /**
   * Extract the user's actual PDS service URL from the DID document.
   */
  _extractPdsUrl(didDoc) {
    if (!didDoc?.service) return;
    const pdsSvc = didDoc.service.find(s =>
      s.id === '#atproto_pds' || s.type === 'AtprotoPersonalDataServer'
    );
    if (pdsSvc?.serviceEndpoint) {
      this.pdsUrl = pdsSvc.serviceEndpoint.replace(/\/$/, '') + '/xrpc';
    }
  }

  /**
   * Refresh the access token.
   */
  async refreshSession() {
    if (!this.session?.refreshJwt) throw new Error('No refresh token');
    const res = await fetch(`${BSKY_API}/com.atproto.server.refreshSession`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.session.refreshJwt}` }
    });
    if (!res.ok) throw new Error(`Refresh failed: ${res.status}`);
    const data = await res.json();
    this.session = {
      did: data.did,
      handle: data.handle,
      accessJwt: data.accessJwt,
      refreshJwt: data.refreshJwt
    };
    if (data.didDoc) this._extractPdsUrl(data.didDoc);
    this.onSessionChange?.(this.session);
    return this.session;
  }

  /**
   * Create a post with optional image.
   */
  async createPost(text, imageBlob = null, altText = '') {
    const record = {
      $type: 'app.bsky.feed.post',
      text,
      createdAt: new Date().toISOString()
    };

    // Add link facets for any URLs in text
    const facets = detectLinkFacets(text);
    if (facets.length > 0) record.facets = facets;

    // Attach image if provided
    if (imageBlob) {
      const blobRef = await this.uploadBlob(imageBlob);
      record.embed = {
        $type: 'app.bsky.embed.images',
        images: [{
          alt: altText || text.slice(0, 100),
          image: blobRef,
          aspectRatio: { width: 1, height: 1 }
        }]
      };
    }

    return await this._fetch('com.atproto.repo.createRecord', {
      method: 'POST',
      body: {
        repo: this.session.did,
        collection: 'app.bsky.feed.post',
        record
      }
    });
  }

  /**
   * Reply to a post with optional image.
   */
  async replyToPost(parentUri, parentCid, rootUri, rootCid, text, imageBlob = null, altText = '') {
    const record = {
      $type: 'app.bsky.feed.post',
      text,
      createdAt: new Date().toISOString(),
      reply: {
        root: { uri: rootUri || parentUri, cid: rootCid || parentCid },
        parent: { uri: parentUri, cid: parentCid }
      }
    };

    // Add link facets for any URLs in text
    const replyFacets = detectLinkFacets(text);
    if (replyFacets.length > 0) record.facets = replyFacets;

    if (imageBlob) {
      const blobRef = await this.uploadBlob(imageBlob);
      record.embed = {
        $type: 'app.bsky.embed.images',
        images: [{
          alt: altText || text.slice(0, 100),
          image: blobRef,
          aspectRatio: { width: 1, height: 1 }
        }]
      };
    }

    return await this._fetch('com.atproto.repo.createRecord', {
      method: 'POST',
      body: {
        repo: this.session.did,
        collection: 'app.bsky.feed.post',
        record
      }
    });
  }

  /**
   * Upload an image blob.
   */
  async uploadBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const res = await fetch(`${BSKY_API}/com.atproto.repo.uploadBlob`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.session.accessJwt}`,
        'Content-Type': blob.type || 'image/png'
      },
      body: arrayBuffer
    });

    if (res.status === 401) {
      await this.refreshSession();
      return this.uploadBlob(blob);
    }

    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    const data = await res.json();
    return data.blob;
  }

  /**
   * Get notifications (replies, mentions, likes).
   */
  async getNotifications(limit = 30) {
    const params = new URLSearchParams({ limit: String(limit) });
    return await this._fetch(`app.bsky.notification.listNotifications?${params}`);
  }

  /**
   * Get own profile.
   */
  async getProfile() {
    const params = new URLSearchParams({ actor: this.session.did });
    return await this._fetch(`app.bsky.actor.getProfile?${params}`);
  }

  /**
   * Get own recent posts.
   */
  async getAuthorFeed(limit = 5) {
    const params = new URLSearchParams({
      actor: this.session.did,
      limit: String(limit)
    });
    return await this._fetch(`app.bsky.feed.getAuthorFeed?${params}`);
  }

  /**
   * Get a specific post thread (to find root for replies).
   */
  async getPostThread(uri, depth = 0) {
    const params = new URLSearchParams({ uri, depth: String(depth) });
    return await this._fetch(`app.bsky.feed.getPostThread?${params}`);
  }

  /**
   * Mark notifications as seen up to the current time.
   */
  async updateSeenNotifications() {
    return await this._fetch('app.bsky.notification.updateSeen', {
      method: 'POST',
      body: { seenAt: new Date().toISOString() }
    });
  }

  /**
   * Follow a user by DID.
   */
  async followUser(did) {
    return await this._fetch('com.atproto.repo.createRecord', {
      method: 'POST',
      body: {
        repo: this.session.did,
        collection: 'app.bsky.graph.follow',
        record: {
          $type: 'app.bsky.graph.follow',
          subject: did,
          createdAt: new Date().toISOString()
        }
      }
    });
  }

  /**
   * Get followers who we are not following back.
   * Returns array of { did, handle, displayName }.
   */
  async getUnfollowedFollowers() {
    // Get our followers
    const followersRes = await this._fetch(
      `app.bsky.graph.getFollowers?actor=${this.session.did}&limit=100`
    );
    const followers = (followersRes.followers || []).map(f => ({
      did: f.did, handle: f.handle, displayName: f.displayName || ''
    }));

    // Get who we follow
    const followsRes = await this._fetch(
      `app.bsky.graph.getFollows?actor=${this.session.did}&limit=100`
    );
    const followingDids = new Set((followsRes.follows || []).map(f => f.did));

    // Return followers we don't follow back (skip invalid handles)
    return followers.filter(f => !followingDids.has(f.did) && f.handle !== 'handle.invalid');
  }

  /**
   * Get thread context for a post (parent chain up to 3 levels).
   * Returns an array of { author, text } from oldest to newest.
   */
  async getThreadContext(uri) {
    try {
      const thread = await this.getPostThread(uri, 3);
      const context = [];
      let node = thread.thread;
      // Walk up the parent chain
      const stack = [];
      while (node) {
        if (node.post?.record?.text) {
          stack.push({
            author: node.post.author?.handle || 'unknown',
            text: node.post.record.text
          });
        }
        node = node.parent;
      }
      // Reverse so oldest is first
      return stack.reverse();
    } catch {
      return [];
    }
  }

  /**
   * Update profile description to include ETH address.
   */
  async updateProfileDescription(description) {
    // First get current profile record
    const params = new URLSearchParams({
      repo: this.session.did,
      collection: 'app.bsky.actor.profile',
      rkey: 'self'
    });

    let existing = {};
    try {
      const current = await this._fetch(`com.atproto.repo.getRecord?${params}`);
      existing = current.value || {};
    } catch {
      // No existing profile record
    }

    // Update description, preserve other fields
    const record = {
      ...existing,
      $type: 'app.bsky.actor.profile',
      description
    };

    return await this._fetch('com.atproto.repo.putRecord', {
      method: 'POST',
      body: {
        repo: this.session.did,
        collection: 'app.bsky.actor.profile',
        rkey: 'self',
        record
      }
    });
  }

  // ─── Chat / Direct Messages ──────────────────────────────────────

  /**
   * List conversations (DMs).
   */
  async listConversations(limit = 30, cursor = '') {
    const params = new URLSearchParams({ limit: String(limit) });
    if (cursor) params.set('cursor', cursor);
    return await this._fetchChat(`chat.bsky.convo.listConvos?${params}`);
  }

  /**
   * Get messages from a specific conversation.
   */
  async getConvoMessages(convoId, limit = 30, cursor = '') {
    const params = new URLSearchParams({ convoId, limit: String(limit) });
    if (cursor) params.set('cursor', cursor);
    return await this._fetchChat(`chat.bsky.convo.getMessages?${params}`);
  }

  /**
   * Send a text message to a conversation.
   */
  async sendChatMessage(convoId, text) {
    return await this._fetchChat('chat.bsky.convo.sendMessage', {
      method: 'POST',
      body: { convoId, message: { text } }
    });
  }

  /**
   * Mark a conversation as read.
   */
  async markConvoRead(convoId) {
    return await this._fetchChat('chat.bsky.convo.updateRead', {
      method: 'POST',
      body: { convoId }
    });
  }

  /**
   * Internal fetch wrapper for chat endpoints (requires atproto-proxy header).
   */
  async _fetchChat(endpoint, options = {}) {
    const { method = 'GET', body = null } = options;
    const url = `${this.pdsUrl}/${endpoint}`;

    const headers = {
      'Content-Type': 'application/json',
      'atproto-proxy': 'did:web:api.bsky.chat#bsky_chat'
    };
    if (this.session?.accessJwt) {
      headers['Authorization'] = `Bearer ${this.session.accessJwt}`;
    }

    const fetchOpts = { method, headers };
    if (body) fetchOpts.body = JSON.stringify(body);

    let res = await fetch(url, fetchOpts);

    // Auto-refresh on 401
    if (res.status === 401 && this.session?.refreshJwt) {
      await this.refreshSession();
      headers['Authorization'] = `Bearer ${this.session.accessJwt}`;
      res = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined });
    }

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Bluesky ${endpoint.split('?')[0]} ${res.status}: ${errText.slice(0, 300)}`);
    }

    const text = await res.text();
    if (!text) return {};
    try { return JSON.parse(text); } catch { return {}; }
  }

  /**
   * Internal fetch wrapper with auth and retry.
   */
  async _fetch(endpoint, options = {}) {
    const { method = 'GET', body = null, noAuth = false } = options;
    const url = endpoint.startsWith('http') ? endpoint : `${BSKY_API}/${endpoint}`;

    const headers = { 'Content-Type': 'application/json' };
    if (!noAuth && this.session?.accessJwt) {
      headers['Authorization'] = `Bearer ${this.session.accessJwt}`;
    }

    const fetchOpts = { method, headers };
    if (body) fetchOpts.body = JSON.stringify(body);

    let res = await fetch(url, fetchOpts);

    // Auto-refresh on 401
    if (res.status === 401 && !noAuth && this.session?.refreshJwt) {
      await this.refreshSession();
      headers['Authorization'] = `Bearer ${this.session.accessJwt}`;
      res = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined });
    }

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Bluesky ${endpoint.split('?')[0]} ${res.status}: ${errText.slice(0, 300)}`);
    }

    // Some endpoints (e.g. updateSeen) return empty body
    const text = await res.text();
    if (!text) return {};
    try { return JSON.parse(text); } catch { return {}; }
  }
}
