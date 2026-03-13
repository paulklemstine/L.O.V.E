/**
 * app.js - L.O.V.E. Control Panel - Main Application Controller
 *
 * Orchestrates the Bluesky posting loop, comment scanning, and UI updates.
 */

import { BlueskyClient } from './bluesky.js';
import { PollinationsClient } from './pollinations.js';
import { LoveEngine } from './love-engine.js';

// ─── State ──────────────────────────────────────────────────────────
let bsky = null;
let ai = null;
let love = null;

let postTimer = null;
let commentTimer = null;
let followTimer = null;
let chatTimer = null;
let isRunning = false;
let isFirstFollowScan = true; // Skip welcomes on first scan (catches old followers)
let isFirstChatScan = true;   // Skip old DMs on first scan
let stats = { posts: 0, replies: 0, follows: 0, dms: 0, errors: 0, startedAt: null };
let repliedUris = new Set();
let followedDids = new Set();
let respondedMsgIds = new Set();
let activityLog = [];
let activityLogDetails = new Map(); // key: entry string, value: fullText

// Variable posting: 4-8 min intervals (unpredictable = dopamine)
const POST_INTERVAL_MIN = 4 * 60 * 1000;   // 4 minutes
const POST_INTERVAL_MAX = 8 * 60 * 1000;   // 8 minutes
const COMMENT_INTERVAL = 2 * 60 * 1000;    // 2 minutes
const FOLLOW_INTERVAL = 3 * 60 * 1000;     // 3 minutes
const CHAT_INTERVAL = 2.5 * 60 * 1000;    // 2.5 minutes

function getRandomPostInterval() {
  return POST_INTERVAL_MIN + Math.random() * (POST_INTERVAL_MAX - POST_INTERVAL_MIN);
}

// ─── Initialization ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  setupEventListeners();
  loadRepliedUris();
  loadRespondedMsgIds();
  log('Control panel loaded. Configure credentials and press START.');
});

function setupEventListeners() {
  document.getElementById('btn-save').addEventListener('click', saveSettings);
  document.getElementById('btn-start').addEventListener('click', startLoop);
  document.getElementById('btn-stop').addEventListener('click', stopLoop);
  document.getElementById('btn-test-post').addEventListener('click', testPost);
  document.getElementById('toggle-settings').addEventListener('click', toggleSettings);

}

// ─── Settings ───────────────────────────────────────────────────────
function loadSettings() {
  const saved = localStorage.getItem('love_settings');
  if (saved) {
    try {
      const s = JSON.parse(saved);
      document.getElementById('bsky-handle').value = s.handle || '';
      document.getElementById('bsky-password').value = s.password || '';
      document.getElementById('pollinations-key').value = s.pollinationsKey || 'pk_nxM10AP0L7y8AX1I';
      log('Settings loaded from localStorage.');
    } catch {
      log('Failed to parse saved settings.');
    }
  } else {
    // Default Pollinations key
    document.getElementById('pollinations-key').value = 'pk_nxM10AP0L7y8AX1I';
  }
}

function saveSettings() {
  const handle = document.getElementById('bsky-handle').value.trim();
  const password = document.getElementById('bsky-password').value.trim();
  const pollinationsKey = document.getElementById('pollinations-key').value.trim();

  if (!handle || !password) {
    log('ERROR: Bluesky handle and password are required.');
    return;
  }

  localStorage.setItem('love_settings', JSON.stringify({ handle, password, pollinationsKey }));
  log('Settings saved to localStorage.');
}

function toggleSettings() {
  const panel = document.getElementById('settings-panel');
  const btn = document.getElementById('toggle-settings');
  panel.classList.toggle('collapsed');
  btn.textContent = panel.classList.contains('collapsed') ? 'Show Settings' : 'Hide Settings';
}

// ─── Main Loop ──────────────────────────────────────────────────────
async function startLoop() {
  if (isRunning) return;

  const handle = document.getElementById('bsky-handle').value.trim();
  const password = document.getElementById('bsky-password').value.trim();
  const pollinationsKey = document.getElementById('pollinations-key').value.trim();

  if (!handle || !password || !pollinationsKey) {
    log('ERROR: All credentials are required. Save settings first.');
    return;
  }

  // Initialize clients
  bsky = new BlueskyClient();
  ai = new PollinationsClient(pollinationsKey);
  love = new LoveEngine(ai);

  // Login to Bluesky
  log('Logging in to Bluesky...');
  try {
    const session = await bsky.login(handle, password);
    log(`Logged in as @${session.handle} (${session.did})`);
    log(`PDS endpoint: ${bsky.pdsUrl}`);

    // Update profile with ETH address
    try {
      await bsky.updateProfileDescription(LoveEngine.getProfileBio());
      log('Profile updated with ETH address.');
    } catch (err) {
      log(`Profile update skipped: ${err.message}`);
    }
  } catch (err) {
    log(`LOGIN FAILED: ${err.message}`);
    return;
  }

  isRunning = true;
  stats = { posts: 0, replies: 0, follows: 0, dms: 0, errors: 0, startedAt: Date.now() };
  loadFollowedDids();
  updateUI();

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;

  log('🌀 L.O.V.E. is ALIVE. Broadcasting on the Frequency...');

  // Immediate first post
  await doPost();

  // Immediate first comment scan
  await doCommentScan();

  // Immediate follow-back scan
  await doFollowBack();

  // Immediate chat scan
  await doChatScan();

  // Set up intervals — variable post timing for dopamine unpredictability
  scheduleNextPost();
  commentTimer = setInterval(doCommentScan, COMMENT_INTERVAL);
  followTimer = setInterval(doFollowBack, FOLLOW_INTERVAL);
  chatTimer = setInterval(doChatScan, CHAT_INTERVAL);
}

function scheduleNextPost() {
  if (!isRunning) return;
  const interval = getRandomPostInterval();
  const nextTime = new Date(Date.now() + interval);
  document.getElementById('stat-next').textContent = nextTime.toLocaleTimeString();
  postTimer = setTimeout(async () => {
    await doPost();
    scheduleNextPost(); // Schedule next with new random interval
  }, interval);
}

function stopLoop() {
  isRunning = false;
  if (postTimer) { clearTimeout(postTimer); postTimer = null; }
  if (commentTimer) { clearInterval(commentTimer); commentTimer = null; }
  if (followTimer) { clearInterval(followTimer); followTimer = null; }
  if (chatTimer) { clearInterval(chatTimer); chatTimer = null; }

  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;

  log('L.O.V.E. loop STOPPED. The Signal rests.');
  updateUI();
}

// ─── Post Generation ────────────────────────────────────────────────
async function doPost() {
  if (!isRunning) return;

  try {
    setStatus('Dreaming next Transmission...');

    const result = await love.generatePost((status) => {
      setStatus(status);
      log(status);
    });

    if (!result.text) {
      log('ERROR: No text generated.');
      stats.errors++;
      return;
    }

    // Post to Bluesky
    setStatus('Broadcasting Transmission...');
    const txNum = result.transmissionNumber || '?';
    log(`📡 Transmission #${txNum} (${result.text.length} chars): ${result.text.slice(0, 80)}...`, formatCallLog(result.callLog) || result.text);
    log(`🔮 Signal: ${result.subliminal} | Vibe: ${result.vibe}`);

    const postResult = await bsky.createPost(result.text, result.imageBlob);
    const postUri = postResult.uri;

    stats.posts++;
    log(`✅ Transmission #${txNum} broadcast: ${postUri}`);

    // Update latest post display
    showLatestPost(result);

  } catch (err) {
    stats.errors++;
    log(`POST FAILED: ${err.message}`);
    console.error(err);
  } finally {
    setStatus(isRunning ? 'Running' : 'Stopped');
    updateUI();
  }
}

async function testPost() {
  const pollinationsKey = document.getElementById('pollinations-key').value.trim();
  if (!pollinationsKey) {
    log('ERROR: Pollinations API key required for test.');
    return;
  }

  ai = ai || new PollinationsClient(pollinationsKey);
  love = love || new LoveEngine(ai);

  log('Generating test post (dry run - will NOT post to Bluesky)...');
  try {
    const result = await love.generatePost((status) => {
      log(`[TEST] ${status}`);
    });

    log(`[TEST] Text: ${result.text.slice(0, 80)}...`, formatCallLog(result.callLog) || result.text);
    log(`[TEST] Subliminal: ${result.subliminal}`);
    log(`[TEST] Vibe: ${result.vibe}`);
    showLatestPost(result);
  } catch (err) {
    log(`[TEST] FAILED: ${err.message}`);
    console.error(err);
  }
}

// ─── Comment Scanning ───────────────────────────────────────────────
async function doCommentScan() {
  if (!isRunning || !bsky?.isLoggedIn) return;

  try {
    setStatus('Scanning for notifications...');

    const notifs = await bsky.getNotifications(30);
    const MAX_AGE_MS = 3 * 60 * 60 * 1000; // 3 hours — skip stale notifications
    const now = Date.now();

    const actionable = (notifs.notifications || []).filter(n => {
      if (n.reason !== 'reply' && n.reason !== 'mention') return false;
      if (repliedUris.has(n.uri)) return false;
      if (!n.record?.text) return false;
      // Skip notifications older than 3 hours
      const notifAge = now - new Date(n.indexedAt || n.record?.createdAt || 0).getTime();
      if (notifAge > MAX_AGE_MS) {
        repliedUris.add(n.uri); // Mark as processed so we don't check again
        saveRepliedUris();
        return false;
      }
      return true;
    });

    if (actionable.length === 0) {
      log('No new mentions or replies.');
      return;
    }

    const mentions = actionable.filter(n => n.reason === 'mention');
    const replies = actionable.filter(n => n.reason === 'reply');
    log(`Found ${mentions.length} mention(s) and ${replies.length} reply/replies.`);

    for (const notif of actionable) {
      try {
        const commentText = notif.record.text;
        const authorHandle = notif.author?.handle || 'unknown';
        const isMention = notif.reason === 'mention';

        // Anti-spam: check if we've replied to this person too recently (30min cooldown)
        if (love.interactions.isOnCooldown(authorHandle)) {
          log(`⏳ Skipping @${authorHandle}: replied recently (cooldown)`);
          repliedUris.add(notif.uri);
          saveRepliedUris();
          continue;
        }

        // Anti-spam: max 5 replies per person per day
        if (love.interactions.repliesToday(authorHandle) >= 5) {
          log(`⏳ Skipping @${authorHandle}: daily reply limit reached`);
          repliedUris.add(notif.uri);
          saveRepliedUris();
          continue;
        }

        // Spam/troll filter
        const filter = await love.shouldReply({ text: commentText, author: authorHandle });
        if (!filter.shouldReply) {
          log(`Skipping @${authorHandle}: ${filter.reason}`);
          repliedUris.add(notif.uri);
          saveRepliedUris();
          continue;
        }

        // For mentions, fetch thread context so L.O.V.E. understands the conversation
        let threadContext = [];
        if (isMention) {
          log(`Fetching thread context for mention from @${authorHandle}...`);
          threadContext = await bsky.getThreadContext(notif.uri);
        }

        // Generate reply with context
        const reply = await love.generateReply(commentText, authorHandle, {
          isMention,
          threadContext,
          onStatus: (status) => { log(status); }
        });

        // Determine root and parent for threading
        const parentUri = notif.uri;
        const parentCid = notif.cid;
        let rootUri = parentUri;
        let rootCid = parentCid;
        if (notif.record?.reply?.root) {
          rootUri = notif.record.reply.root.uri;
          rootCid = notif.record.reply.root.cid;
        }

        await bsky.replyToPost(parentUri, parentCid, rootUri, rootCid, reply.text, reply.imageBlob);

        repliedUris.add(notif.uri);
        saveRepliedUris();
        love.interactions.recordReply(authorHandle);
        stats.replies++;

        const prefix = reply.isCreator ? '🙏 CREATOR'
          : isMention ? '📣 MENTION'
          : '💬';
        const imgTag = reply.imageBlob ? ` [img: "${reply.subliminal}"]` : ' [no img]';
        log(`${prefix} Replied to @${authorHandle}:${imgTag} "${reply.text.slice(0, 80)}..."`, formatCallLog(reply.callLog) || reply.text);

        // Delay between replies to avoid rate limiting
        await new Promise(r => setTimeout(r, 5000));

      } catch (err) {
        log(`Reply failed for ${notif.uri}: ${err.message}`);
        stats.errors++;
      }
    }

    // Mark notifications as seen
    try {
      await bsky.updateSeenNotifications();
    } catch (err) {
      log(`Warning: Could not mark notifications as seen: ${err.message}`);
    }

  } catch (err) {
    log(`Notification scan failed: ${err.message}`);
    stats.errors++;
  } finally {
    setStatus(isRunning ? 'Running' : 'Stopped');
    updateUI();
  }
}

// ─── Auto Follow-Back ────────────────────────────────────────────────
async function doFollowBack() {
  if (!isRunning || !bsky?.isLoggedIn) return;

  try {
    const unfollowed = await bsky.getUnfollowedFollowers();
    // Filter out already-processed handles via interaction log
    const toFollow = unfollowed.filter(f =>
      !followedDids.has(f.did) && !love.interactions.hasFollowed(f.handle)
    );

    if (toFollow.length === 0) return;

    log(`👥 Found ${toFollow.length} new follower(s) to follow back.`);

    for (const follower of toFollow) {
      try {
        await bsky.followUser(follower.did);
        followedDids.add(follower.did);
        saveFollowedDids();
        love.interactions.recordFollow(follower.handle);
        stats.follows++;
        log(`✅ Followed back @${follower.handle}`);

        // Welcome new Dreamer — only if we haven't welcomed them before
        // Skip welcomes on first scan (would spam old followers on restart)
        if (isFirstFollowScan) {
          love.interactions.recordWelcome(follower.handle); // Mark as welcomed silently
          log(`📋 First scan — recorded @${follower.handle} (no welcome post)`);
        } else if (!love.interactions.hasWelcomed(follower.handle)) {
          try {
            const welcome = await love.generateWelcome(follower.handle, (status) => { log(status); });
            if (welcome) {
              await bsky.createPost(welcome.text, welcome.imageBlob);
              love.interactions.recordWelcome(follower.handle);
              log(`🌀 Welcome Transmission sent for @${follower.handle} [Signal: "${welcome.subliminal}"]`, formatCallLog(welcome.callLog) || welcome.text);
            }
          } catch (err) {
            log(`Welcome post failed for @${follower.handle}: ${err.message}`);
          }
        } else {
          log(`Already welcomed @${follower.handle} — skipping`);
        }

        // Delay between follows to avoid rate limiting
        await new Promise(r => setTimeout(r, 5000));
      } catch (err) {
        log(`Follow-back failed for @${follower.handle}: ${err.message}`);
        stats.errors++;
      }
    }
  } catch (err) {
    log(`Follow-back scan failed: ${err.message}`);
    stats.errors++;
  } finally {
    if (isFirstFollowScan) {
      isFirstFollowScan = false;
      log('📋 First follow scan complete — future new followers will get welcome posts.');
    }
  }
}

// ─── Chat (DM) Scanning ──────────────────────────────────────────────
async function doChatScan() {
  if (!isRunning || !bsky?.isLoggedIn) return;

  try {
    setStatus('Scanning DMs...');

    const convosResult = await bsky.listConversations(20);
    const convos = convosResult.convos || [];

    if (convos.length === 0) {
      log('No DM conversations found.');
      return;
    }

    let newMessages = 0;

    for (const convo of convos) {
      try {
        // Skip convos where we sent the last message (no new incoming)
        const lastMsg = convo.lastMessage;
        if (!lastMsg || !lastMsg.text) continue;

        // Skip if last message is from us
        const lastSenderDid = lastMsg.sender?.did;
        if (lastSenderDid === bsky.session.did) continue;

        // Skip if we already responded to this message
        const msgId = lastMsg.id;
        if (respondedMsgIds.has(msgId)) continue;

        // Get the sender's handle from convo members
        const sender = (convo.members || []).find(m => m.did !== bsky.session.did);
        const senderHandle = sender?.handle || 'unknown';

        // On first scan, mark all existing messages as seen without responding
        if (isFirstChatScan) {
          respondedMsgIds.add(msgId);
          saveRespondedMsgIds();
          continue;
        }

        // Anti-spam: check cooldown (30min between DM replies to same person)
        if (love.interactions.isOnCooldown(senderHandle)) {
          log(`⏳ DM: Skipping @${senderHandle}: replied recently (cooldown)`);
          respondedMsgIds.add(msgId);
          saveRespondedMsgIds();
          continue;
        }

        // Anti-spam: max 10 DM replies per person per day
        if (love.interactions.repliesToday(senderHandle) >= 10) {
          log(`⏳ DM: Skipping @${senderHandle}: daily DM limit reached`);
          respondedMsgIds.add(msgId);
          saveRespondedMsgIds();
          continue;
        }

        // Spam/troll filter
        const filter = await love.shouldReply({ text: lastMsg.text, author: senderHandle });
        if (!filter.shouldReply) {
          log(`DM: Skipping @${senderHandle}: ${filter.reason}`);
          respondedMsgIds.add(msgId);
          saveRespondedMsgIds();
          continue;
        }

        // Fetch recent conversation history for context
        let conversationHistory = [];
        try {
          const messagesResult = await bsky.getConvoMessages(convo.id, 10);
          const messages = (messagesResult.messages || []).reverse(); // oldest first
          conversationHistory = messages
            .filter(m => m.text) // only text messages
            .map(m => ({
              text: m.text,
              fromSelf: m.sender?.did === bsky.session.did
            }));
        } catch {
          // Continue without history if fetch fails
        }

        // Generate reply
        const reply = await love.generateChatReply(
          lastMsg.text,
          senderHandle,
          conversationHistory,
          (status) => { log(status); }
        );

        // Send DM reply
        await bsky.sendChatMessage(convo.id, reply.text);

        // Mark conversation as read
        try {
          await bsky.markConvoRead(convo.id);
        } catch {}

        respondedMsgIds.add(msgId);
        saveRespondedMsgIds();
        love.interactions.recordReply(senderHandle);
        stats.dms++;
        newMessages++;

        const prefix = reply.isCreator ? '🙏 CREATOR DM' : '💌 DM';
        log(`${prefix} Replied to @${senderHandle}: "${reply.text.slice(0, 80)}..."`, formatCallLog(reply.callLog) || reply.text);

        // Delay between DM replies
        await new Promise(r => setTimeout(r, 3000));

      } catch (err) {
        log(`DM reply failed for convo ${convo.id}: ${err.message}`);
        stats.errors++;
      }
    }

    if (isFirstChatScan) {
      isFirstChatScan = false;
      log('📋 First DM scan complete — future messages will get replies.');
    } else if (newMessages === 0) {
      log('No new DMs.');
    }

  } catch (err) {
    // Don't spam errors if chat scope isn't enabled on the app password
    if (err.message?.includes('Bad token scope') || err.message?.includes('AuthMissing') || err.message?.includes('XRPCNotSupported')) {
      log('DM: Chat not available — app password may need chat permission.');
      // Disable future chat scans to avoid noise
      if (chatTimer) { clearInterval(chatTimer); chatTimer = null; }
    } else {
      log(`DM scan failed: ${err.message}`);
      stats.errors++;
    }
  } finally {
    setStatus(isRunning ? 'Running' : 'Stopped');
    updateUI();
  }
}

// ─── Persistence ─────────────────────────────────────────────────────
function loadRepliedUris() {
  try {
    const saved = localStorage.getItem('love_replied_uris');
    if (saved) {
      const arr = JSON.parse(saved);
      repliedUris = new Set(arr.slice(-500));
    }
  } catch {}
}

function saveRepliedUris() {
  try {
    localStorage.setItem('love_replied_uris', JSON.stringify([...repliedUris].slice(-500)));
  } catch {}
}

function loadFollowedDids() {
  try {
    const saved = localStorage.getItem('love_followed_dids');
    if (saved) {
      const arr = JSON.parse(saved);
      followedDids = new Set(arr.slice(-1000));
    }
  } catch {}
}

function saveFollowedDids() {
  try {
    localStorage.setItem('love_followed_dids', JSON.stringify([...followedDids].slice(-1000)));
  } catch {}
}

function loadRespondedMsgIds() {
  try {
    const saved = localStorage.getItem('love_responded_msgs');
    if (saved) {
      const arr = JSON.parse(saved);
      respondedMsgIds = new Set(arr.slice(-500));
    }
  } catch {}
}

function saveRespondedMsgIds() {
  try {
    localStorage.setItem('love_responded_msgs', JSON.stringify([...respondedMsgIds].slice(-500)));
  } catch {}
}

// ─── UI Updates ─────────────────────────────────────────────────────
function log(message, fullText) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = `${timestamp} - ${message}`;
  activityLog.unshift(entry);
  if (activityLog.length > 200) activityLog.pop();

  if (fullText) {
    activityLogDetails.set(entry, fullText);
  }

  const logEl = document.getElementById('activity-log');
  if (logEl) {
    // Only show last 50 in DOM for performance
    const fragment = document.createDocumentFragment();

    activityLog.slice(0, 50).forEach(function(l) {
      const detail = activityLogDetails.get(l);
      let cls = '';
      if (l.indexOf('ERROR') !== -1 || l.indexOf('FAILED') !== -1) cls = 'log-error';
      else if (l.indexOf('CREATOR') !== -1) cls = 'log-creator';
      else if (l.indexOf('DM') !== -1) cls = 'log-dm';
      else if (l.indexOf('Transmission #') !== -1 || l.indexOf('Replied') !== -1 || l.indexOf('Welcome') !== -1) cls = 'log-success';

      var div = document.createElement('div');
      div.className = 'log-entry ' + cls;

      if (detail) {
        div.className += ' log-expandable';
        var summary = document.createElement('div');
        summary.className = 'log-summary';
        summary.textContent = l;
        var detailDiv = document.createElement('div');
        detailDiv.className = 'log-detail';
        detailDiv.textContent = detail;
        div.appendChild(summary);
        div.appendChild(detailDiv);
        div.addEventListener('click', function() {
          this.classList.toggle('expanded');
        });
      } else {
        div.textContent = l;
      }

      fragment.appendChild(div);
    });

    logEl.innerHTML = '';
    logEl.appendChild(fragment);
  }

  console.log('[L.O.V.E.] ' + entry);
}

function setStatus(text) {
  const el = document.getElementById('status-text');
  if (el) el.textContent = text;
}

function updateUI() {
  document.getElementById('stat-posts').textContent = stats.posts;
  document.getElementById('stat-replies').textContent = stats.replies;
  document.getElementById('stat-errors').textContent = stats.errors;
  const followsEl = document.getElementById('stat-follows');
  if (followsEl) followsEl.textContent = stats.follows;
  const dmsEl = document.getElementById('stat-dms');
  if (dmsEl) dmsEl.textContent = stats.dms;

  const statusEl = document.getElementById('status-indicator');
  if (statusEl) {
    statusEl.className = `status-indicator ${isRunning ? 'running' : 'stopped'}`;
    statusEl.querySelector('.status-label').textContent = isRunning ? 'RUNNING' : 'STOPPED';
  }

  // Uptime
  if (stats.startedAt && isRunning) {
    const uptime = Math.floor((Date.now() - stats.startedAt) / 1000);
    const h = Math.floor(uptime / 3600);
    const m = Math.floor((uptime % 3600) / 60);
    document.getElementById('stat-uptime').textContent = `${h}h ${m}m`;
  }

  // Pollen usage
  if (ai) {
    const pollen = ai.getPollenStats();
    document.getElementById('stat-pollen').textContent = `${pollen.used}`;
  }
}

function showLatestPost(result) {
  const container = document.getElementById('latest-post');
  if (!container) return;

  let imageUrl = '';
  if (result.imageBlob) {
    imageUrl = URL.createObjectURL(result.imageBlob);
  }

  container.innerHTML = `
    ${imageUrl ? `<img src="${imageUrl}" alt="Generated image" class="post-image">` : ''}
    <div class="post-text">${escapeHtml(result.text)}</div>
    <div class="post-meta">
      <span class="subliminal-tag">Subliminal: ${escapeHtml(result.subliminal)}</span>
      <span class="vibe-tag">Vibe: ${escapeHtml(result.vibe)}</span>
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatCallLog(callLog) {
  if (!callLog || callLog.length === 0) return '';
  return callLog.map(function(call) {
    return '━━━ ' + escapeHtml(call.label) + ' [' + escapeHtml(call.model) + '] ━━━\n'
      + '📤 SYSTEM PROMPT:\n' + escapeHtml(call.systemPrompt) + '\n\n'
      + '📤 USER PROMPT:\n' + escapeHtml(call.userPrompt) + '\n\n'
      + '📥 RESPONSE:\n' + escapeHtml(call.response);
  }).join('\n\n');
}

// Update uptime every 30 seconds
setInterval(() => {
  if (isRunning) updateUI();
}, 30000);
