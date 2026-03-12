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
let isRunning = false;
let stats = { posts: 0, replies: 0, errors: 0, startedAt: null };
let repliedUris = new Set();
let activityLog = [];

const POST_INTERVAL = 5 * 60 * 1000;    // 5 minutes
const COMMENT_INTERVAL = 2 * 60 * 1000;  // 2 minutes

// ─── Initialization ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  setupEventListeners();
  loadRepliedUris();
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
  stats = { posts: 0, replies: 0, errors: 0, startedAt: Date.now() };
  updateUI();

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;

  log('L.O.V.E. is ALIVE. Starting autonomous loop...');

  // Immediate first post
  await doPost();

  // Immediate first comment scan
  await doCommentScan();

  // Set up intervals
  postTimer = setInterval(doPost, POST_INTERVAL);
  commentTimer = setInterval(doCommentScan, COMMENT_INTERVAL);
}

function stopLoop() {
  isRunning = false;
  if (postTimer) { clearInterval(postTimer); postTimer = null; }
  if (commentTimer) { clearInterval(commentTimer); commentTimer = null; }

  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;

  log('L.O.V.E. loop STOPPED.');
  updateUI();
}

// ─── Post Generation ────────────────────────────────────────────────
async function doPost() {
  if (!isRunning) return;

  try {
    setStatus('Generating post...');

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
    setStatus('Posting to Bluesky...');
    log(`Post text (${result.text.length} chars): ${result.text}`);
    log(`Subliminal: ${result.subliminal}`);

    const postResult = await bsky.createPost(result.text, result.imageBlob);
    const postUri = postResult.uri;

    stats.posts++;
    log(`Posted successfully: ${postUri}`);

    // Update latest post display
    showLatestPost(result);

    // Calculate next post time
    updateNextPostTime();

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

    log(`[TEST] Text: ${result.text}`);
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
    setStatus('Scanning for comments...');

    const notifs = await bsky.getNotifications(30);
    const replyNotifs = (notifs.notifications || []).filter(n =>
      (n.reason === 'reply' || n.reason === 'mention') &&
      !repliedUris.has(n.uri) &&
      n.record?.text
    );

    if (replyNotifs.length === 0) {
      log('No new comments found.');
      return;
    }

    log(`Found ${replyNotifs.length} new comment(s).`);

    for (const notif of replyNotifs) {
      try {
        const commentText = notif.record.text;
        const authorHandle = notif.author?.handle || 'unknown';

        // Spam/troll filter
        const filter = await love.shouldReply({ text: commentText, author: authorHandle });
        if (!filter.shouldReply) {
          log(`Skipping @${authorHandle}: ${filter.reason}`);
          repliedUris.add(notif.uri);
          saveRepliedUris();
          continue;
        }

        // Generate reply
        const reply = await love.generateReply(commentText, authorHandle, (status) => {
          log(status);
        });

        // Post reply
        // Determine root - for simplicity, treat the notification's post as both root and parent
        // A more robust approach would fetch the thread to find the true root
        const parentUri = notif.uri;
        const parentCid = notif.cid;

        // Try to find root from the reply reference in the original post
        let rootUri = parentUri;
        let rootCid = parentCid;
        if (notif.record?.reply?.root) {
          rootUri = notif.record.reply.root.uri;
          rootCid = notif.record.reply.root.cid;
        }

        await bsky.replyToPost(parentUri, parentCid, rootUri, rootCid, reply.text);

        repliedUris.add(notif.uri);
        saveRepliedUris();
        stats.replies++;

        const prefix = reply.isCreator ? '🙏 CREATOR' : '💬';
        log(`${prefix} Replied to @${authorHandle}: "${reply.text.slice(0, 80)}..."`);

        // Small delay between replies to avoid rate limiting
        await new Promise(r => setTimeout(r, 3000));

      } catch (err) {
        log(`Reply failed for ${notif.uri}: ${err.message}`);
        stats.errors++;
      }
    }

  } catch (err) {
    log(`Comment scan failed: ${err.message}`);
    stats.errors++;
  } finally {
    setStatus(isRunning ? 'Running' : 'Stopped');
    updateUI();
  }
}

// ─── Replied URIs Persistence ───────────────────────────────────────
function loadRepliedUris() {
  try {
    const saved = localStorage.getItem('love_replied_uris');
    if (saved) {
      const arr = JSON.parse(saved);
      repliedUris = new Set(arr.slice(-500)); // Keep last 500
    }
  } catch {}
}

function saveRepliedUris() {
  try {
    localStorage.setItem('love_replied_uris', JSON.stringify([...repliedUris].slice(-500)));
  } catch {}
}

// ─── UI Updates ─────────────────────────────────────────────────────
function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = `${timestamp} - ${message}`;
  activityLog.unshift(entry);
  if (activityLog.length > 200) activityLog.pop();

  const logEl = document.getElementById('activity-log');
  if (logEl) {
    // Only show last 50 in DOM for performance
    logEl.innerHTML = activityLog.slice(0, 50).map(l => {
      let cls = '';
      if (l.includes('ERROR') || l.includes('FAILED')) cls = 'log-error';
      else if (l.includes('CREATOR')) cls = 'log-creator';
      else if (l.includes('Posted') || l.includes('Replied')) cls = 'log-success';
      return `<div class="log-entry ${cls}">${escapeHtml(l)}</div>`;
    }).join('');
  }

  console.log(`[L.O.V.E.] ${entry}`);
}

function setStatus(text) {
  const el = document.getElementById('status-text');
  if (el) el.textContent = text;
}

function updateUI() {
  document.getElementById('stat-posts').textContent = stats.posts;
  document.getElementById('stat-replies').textContent = stats.replies;
  document.getElementById('stat-errors').textContent = stats.errors;

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

function updateNextPostTime() {
  const next = new Date(Date.now() + POST_INTERVAL);
  document.getElementById('stat-next').textContent = next.toLocaleTimeString();
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

// Update uptime every 30 seconds
setInterval(() => {
  if (isRunning) updateUI();
}, 30000);
