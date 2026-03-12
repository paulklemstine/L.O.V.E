/**
 * bluesky.js - Bluesky AT Protocol client for browser
 */

const BSKY_API = 'https://bsky.social/xrpc';

export class BlueskyClient {
  constructor() {
    this.session = null; // { did, handle, accessJwt, refreshJwt }
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
    this.onSessionChange?.(this.session);
    return this.session;
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
  async replyToPost(parentUri, parentCid, rootUri, rootCid, text, imageBlob = null) {
    const record = {
      $type: 'app.bsky.feed.post',
      text,
      createdAt: new Date().toISOString(),
      reply: {
        root: { uri: rootUri || parentUri, cid: rootCid || parentCid },
        parent: { uri: parentUri, cid: parentCid }
      }
    };

    if (imageBlob) {
      const blobRef = await this.uploadBlob(imageBlob);
      record.embed = {
        $type: 'app.bsky.embed.images',
        images: [{
          alt: text.slice(0, 100),
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
