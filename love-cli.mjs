#!/usr/bin/env node
/**
 * love-cli.mjs — Linux CLI port of the L.O.V.E webapp.
 *
 * Runs the same LoveEngine pipeline (love-engine.js) but backed by:
 *   - Local LLM: Ollama (OpenAI-compatible endpoint on 127.0.0.1:11434)
 *   - Local image: SDXL via ai/generate_image.py and ai/render_batch.py
 *   - Bluesky posting via bluesky.js
 *
 * Strictly sequential: one post at a time, text → image → published → next,
 * repeating forever until interrupted. There is no pause between successful
 * posts -- the SDXL render is the natural spacing between them.
 *
 * Usage:
 *   node love-cli.mjs                 # dry run: generate, save image to ./output, print
 *   node love-cli.mjs --post          # generate AND publish to Bluesky, forever
 *   node love-cli.mjs --post --once   # a single post, then exit
 *   node love-cli.mjs --skip-image    # text-only generation (fast)
 *
 * Credentials: read from .env (BLUESKY_HANDLE, BLUESKY_APP_PASSWORD) or env vars.
 */

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { LoveEngine } from "./public/js/love-engine.js";
import { BlueskyClient } from "./public/js/bluesky.js";
import { Backoff } from "./loop.mjs";

// ─── localStorage shim (file-backed, same keys as the webapp) ───────────
const STATE_FILE = path.join(import.meta.dirname, ".love-state.json");
const state = fs.existsSync(STATE_FILE)
    ? JSON.parse(fs.readFileSync(STATE_FILE, "utf8"))
    : {};
globalThis.localStorage = {
    getItem: (k) => (k in state ? state[k] : null),
    setItem: (k, v) => {
        state[k] = String(v);
        fs.writeFileSync(STATE_FILE, JSON.stringify(state));
    },
};

// ─── .env loader (never committed) ──────────────────────────────────────
const ENV_FILE = path.join(import.meta.dirname, ".env");
if (fs.existsSync(ENV_FILE)) {
    for (const line of fs.readFileSync(ENV_FILE, "utf8").split("\n")) {
        const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
        if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
}

const OLLAMA_URL = "http://127.0.0.1:11434";
const OLLAMA_MODEL = "qwen3:8b";
// Image scripts are versioned in this repo (ai/); the venv and model weights
// (~25GB) stay in ~/ai and are located via LOVE_AI_HOME.
const AI_HOME = process.env.LOVE_AI_HOME || path.join(process.env.HOME, "ai");
const AI_SCRIPTS = path.join(import.meta.dirname, "ai");
const IMG_PY = path.join(AI_HOME, "imgenv", "bin", "python");
const IMG_SCRIPT = path.join(AI_SCRIPTS, "generate_image.py");
const OUTPUT_DIR = path.join(import.meta.dirname, "output");

// ─── Local client: same interface as PollinationsClient ─────────────────
class LocalClient {
    constructor(queueOnly = false) {
        this.callLog = [];
        this.queueOnly = queueOnly;
        this.queue = [];
    }
    resetCallLog() { this.callLog = []; }
    getCallLog() { return this.callLog; }

    async generateText(systemPrompt, userPrompt, options = {}) {
        const {
            temperature = 0.85,
            maxRetries = 2,
            label = "LLM Call",
            // qwen3 at high temperature occasionally falls into a repetition loop and
            // generates until the context window is full; without a cap the request can
            // spin for minutes and never return.
            maxTokens = 1024,
            // A stalled request never rejects on its own, so the retry loop below only
            // ever sees errors. Abort it ourselves so a hang becomes a retryable failure.
            timeoutMs = 120000,
        } = options;
        const body = {
            model: OLLAMA_MODEL,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt },
            ],
            // qwen3 defaults to emitting a long <think> block first. On the OpenAI-compatible
            // /v1 endpoint that reasoning is returned in a separate `reasoning` field, leaving
            // `content` empty, and it burns the whole token budget before anything usable comes
            // back. The native endpoint's `think: false` suppresses it outright.
            think: false,
            options: { temperature, num_predict: maxTokens },
            stream: false,
        };
        if (userPrompt.includes("Return ONLY valid JSON") || userPrompt.includes("Return ONLY raw JSON")) {
            body.format = "json";
        }

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const res = await fetch(`${OLLAMA_URL}/api/chat`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                    signal: AbortSignal.timeout(timeoutMs),
                });
                if (!res.ok) throw new Error(`Ollama ${res.status}: ${(await res.text()).slice(0, 200)}`);
                const data = await res.json();
                const text = data.message?.content || "";
                if (!text.trim()) throw new Error("Ollama returned empty response");
                this.callLog.push({ label, systemPrompt, userPrompt, response: text, model: OLLAMA_MODEL });
                return text;
            } catch (err) {
                if (err.name === "TimeoutError") {
                    err = new Error(`timed out after ${timeoutMs}ms`);
                }
                if (attempt === maxRetries) throw err;
                await new Promise((r) => setTimeout(r, 5000 * Math.pow(2, attempt)));
            }
        }
    }

    async generateImage(prompt, options = {}) {
        const { width = 1024, height = 1024, negativePrompt = null, seed = null } = options;
        const fullPrompt = `Very erotic and sensual: ${prompt}`;

        if (this.queueOnly) {
            this.queue.push({ prompt: fullPrompt, negative: negativePrompt, w: width, h: height, seed });
            return null;
        }

        // LLM and SDXL cannot share the 6GB GPU: unload the Ollama model first
        await fetch(`${OLLAMA_URL}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: OLLAMA_MODEL, keep_alive: 0 }),
        }).catch(() => {});
        await new Promise((r) => setTimeout(r, 2000));

        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
        const outPath = path.join(OUTPUT_DIR, `love-${Date.now()}.png`);

        const args = ["--prompt", fullPrompt, "--w", String(width), "--h", String(height), "--out", outPath];
        if (negativePrompt) args.push("--negative", negativePrompt);
        if (seed !== null) args.push("--seed", String(seed));

        await new Promise((resolve, reject) => {
            const proc = spawn(IMG_PY, [IMG_SCRIPT, ...args], { stdio: ["ignore", "pipe", "inherit"] });
            proc.on("error", reject);
            proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`image gen exited ${code}`))));
        });

        const buf = fs.readFileSync(outPath);
        this.callLog.push({ label: "Image", response: outPath, model: "sdxl-local" });
        return new Blob([buf], { type: "image/png" });
    }

    async generateVideo() { throw new Error("Video generation is not available in local mode"); }
    async generateMusic() { throw new Error("Music generation is not available in local mode"); }
    async generateAudio() { throw new Error("TTS is not available in local mode"); }

    extractJSON(text) {
        if (!text) return null;
        text = text.trim();
        const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (codeBlockMatch) text = codeBlockMatch[1].trim();
        try { return JSON.parse(text); } catch {}
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) { try { return JSON.parse(jsonMatch[0]); } catch {} }
        return null;
    }
}

// ─── Ollama supervision (started detached if not already up) ───────────
async function ollamaHealthy() {
    try {
        const res = await fetch(`${OLLAMA_URL}/api/version`, { signal: AbortSignal.timeout(2000) });
        return res.ok;
    } catch {
        return false;
    }
}

async function ensureOllama() {
    if (await ollamaHealthy()) return;
    const startScript = path.join(process.env.HOME, "ai", "start-ollama.sh");
    if (!fs.existsSync(startScript)) throw new Error(`missing ${startScript}`);
    console.error("[llm] Ollama is down, starting it...");
    const child = spawn(startScript, [], { stdio: "ignore", detached: true });
    child.unref();
    for (let i = 0; i < 90; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        if (await ollamaHealthy()) {
            console.error("[llm] Ollama is up");
            return;
        }
    }
    throw new Error("Ollama did not become ready within 90s");
}

// ─── Main ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const doPost = args.includes("--post");
const skipImage = args.includes("--skip-image");
const runOnce = args.includes("--once");

await ensureOllama();

// Bluesky login once, reused across posts
async function getBsky() {
    const handle = process.env.BLUESKY_HANDLE;
    const password = process.env.BLUESKY_APP_PASSWORD;
    if (!handle || !password) {
        console.error("missing BLUESKY_HANDLE / BLUESKY_APP_PASSWORD (set in .env)");
        process.exit(1);
    }
    const bsky = new BlueskyClient();
    await bsky.login(handle, password);
    return bsky;
}

// Bluesky caps blobs at 2MB; re-encode oversized PNGs as JPEG (in place)
async function shrinkForUpload(pngPath) {
    let buf = fs.readFileSync(pngPath);
    if (buf.length > 1_900_000) {
        console.error(`[love] image ${buf.length} bytes, converting to JPEG...`);
        const dst = pngPath.replace(/\.png$/, ".jpg");
        await new Promise((resolve, reject) => {
            const proc = spawn(IMG_PY, ["-c", `
from PIL import Image
im = Image.open("${pngPath}").convert("RGB")
im.thumbnail((1200, 1200))
im.save("${dst}", quality=88)
`]);
            proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error("jpeg conversion failed"))));
        });
        buf = fs.readFileSync(dst);
        return { buf, type: "image/jpeg" };
    }
    return { buf, type: "image/png" };
}

async function postOne(bsky, text, imagePath) {
    const { buf, type } = await shrinkForUpload(imagePath);
    const res = await bsky.createPost(text, new Blob([buf], { type }));
    console.log(`posted: ${res.uri}`);
}

// ─── Follow-back + welcome ───────────────────────────────────────────────
// One scan per post cycle, at the end of the iteration, so a bad scan can
// never disturb the main cadence. This mirrors the webapp's doFollowBack()
// but runs headless -- the browser version only ever runs while the Firebase
// dashboard is open, which meant a continuously-running CLI never followed
// anybody back at all.
//
// Budget is ONE welcome per scan. Each welcome is an LLM call plus a full SDXL
// render (~3-6 min), so an uncapped burst of new followers would block the
// main loop for the length of the whole batch.
//
// Follows are deliberately NOT tracked locally: getUnfollowedFollowers() diffs
// followers against following server-side, so the API stays authoritative and a
// follow that fails simply retries on the next scan. The welcomed record reuses
// engine.interactions, which persists through the localStorage shim above -- no
// second bookkeeping structure to keep in sync.
//
// The pending queue is load-bearing. Following someone back removes them from
// getUnfollowedFollowers(), so a follower whose welcome is deferred by the
// budget would never be seen by a later scan. The queue is what remembers them.
const FOLLOW_BASELINE_KEY = "love_follow_baselined";
const FOLLOW_PENDING_KEY = "love_pending_welcomes";
const FOLLOW_HEARTBEAT_MS = 60 * 60 * 1000;
let lastFollowHeartbeat = 0;

function getPendingWelcomes() {
    try {
        const raw = localStorage.getItem(FOLLOW_PENDING_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr : [];
    } catch {
        return [];
    }
}

async function doFollowBack(bsky, engine, { skipImage = false } = {}) {
    const firstScan = localStorage.getItem(FOLLOW_BASELINE_KEY) !== "true";

    let unfollowed;
    try {
        unfollowed = await bsky.getUnfollowedFollowers();
    } catch (err) {
        console.error(`[follow] scan failed: ${err.message}`);
        return;
    }

    // First run: follow the existing backlog but never welcome it. Posting a
    // welcome to everyone who ever followed, the first time this feature runs,
    // is exactly the burst the webapp guards against with isFirstFollowScan.
    // They are recorded as welcomed so later scans leave them alone.
    if (firstScan) {
        console.error(`[follow] first scan — ${unfollowed.length} follower(s) to follow back (no welcomes this run)`);
        for (const f of unfollowed) {
            try {
                await bsky.followUser(f.did);
                engine.interactions.recordFollow(f.handle);
                engine.interactions.recordWelcome(f.handle);
                console.error(`[follow] baseline: followed @${f.handle} (recorded, not welcomed)`);
                await sleep(5000);
            } catch (err) {
                console.error(`[follow] baseline follow failed for @${f.handle}: ${err.message}`);
            }
        }
        localStorage.setItem(FOLLOW_BASELINE_KEY, "true");
        console.error(`[follow] baseline complete — welcomes begin with the next genuinely new follower`);
        return;
    }

    // Steady state: follow everyone back, queue the ones not yet welcomed.
    const pending = getPendingWelcomes();
    for (const f of unfollowed) {
        try {
            await bsky.followUser(f.did);
            engine.interactions.recordFollow(f.handle);
            console.error(`[follow] followed back @${f.handle}`);
            if (!engine.interactions.hasWelcomed(f.handle) && !pending.includes(f.handle)) {
                pending.push(f.handle);
                console.error(`[follow] queued welcome for @${f.handle}`);
            }
            await sleep(5000);
        } catch (err) {
            console.error(`[follow] follow failed for @${f.handle}: ${err.message}`);
        }
    }
    localStorage.setItem(FOLLOW_PENDING_KEY, JSON.stringify(pending));

    // Drain at most one welcome, if the queue has anything on it. The drain is
    // driven by the queue, not by `unfollowed`: by this point everyone in the
    // queue has already been followed back and so no longer appears in the scan.
    if (pending.length > 0) {
        const handle = pending[0];
        const rest = pending.slice(1);
        try {
            console.error(`[follow] welcoming @${handle}...`);
            const welcome = await engine.generateWelcome(handle, (s) => console.error(`[follow] ${s}`));
            if (welcome?.imageBlob && !skipImage) {
                const safe = handle.replace(/[^\w.-]/g, "_");
                const tmp = path.join(OUTPUT_DIR, `welcome-${safe}.png`);
                fs.mkdirSync(OUTPUT_DIR, { recursive: true });
                fs.writeFileSync(tmp, Buffer.from(await welcome.imageBlob.arrayBuffer()));
                await postOne(bsky, welcome.text, tmp);
            } else if (welcome) {
                const res = await bsky.createPost(welcome.text);
                console.log(`posted: ${res.uri}`);
            }
            engine.interactions.recordWelcome(handle);
            localStorage.setItem(FOLLOW_PENDING_KEY, JSON.stringify(rest));
            if (welcome) {
                console.error(`[follow] welcome posted for @${handle} [Signal: "${welcome.subliminal}"]`);
            } else {
                // generateWelcome returns null for the creator's own handle.
                console.error(`[follow] no welcome generated for @${handle} (excluded by generateWelcome)`);
            }
            if (rest.length > 0) {
                console.error(`[follow] ${rest.length} welcome(s) still queued — one per scan`);
            }
        } catch (err) {
            // The handle stays queued so the next scan retries it.
            console.error(`[follow] welcome failed for @${handle}: ${err.message}`);
            return;
        }
    } else if (unfollowed.length === 0) {
        // Logging this every scan would add a line every ~5 minutes forever.
        const now = Date.now();
        if (now - lastFollowHeartbeat > FOLLOW_HEARTBEAT_MS) {
            lastFollowHeartbeat = now;
            console.error(`[follow] scan: nothing to do`);
        }
    }

    // generateWelcome calls resetCallLog() on the SHARED engine and overwrites
    // lastSubliminalPhrase. Without this reset the next post's "N llm calls"
    // count would be silently wrong.
    engine.ai.resetCallLog();
}

// ── Sequential pipeline: text → image → post, one post at a time, forever ──
// One post is in flight at a time. Each post's image is rendered before the
// next post's text is generated, and is posted before the loop moves on.
//
// This replaces a three-phase design that queued N texts, rendered them all
// in one warm SDXL session, then posted in a burst. That version stalled
// ~50 min before anything reached Bluesky, and lost every render in the
// batch when a process died mid-render.
//
// The trade is a full SDXL model load per image, plus an Ollama unload/reload
// between each post's text and image. Measured cost: 3477s for 10 posts
// (~348s each) against ~50 min for the warm-session batch — about 16% slower,
// not the 2.4x an earlier note here claimed. A render is almost entirely
// diffusion steps, so the ~35s model load is a small fraction of it.
//
// The loop runs until interrupted; there is no batch count. `--once` stops
// after a single post. Consecutive failures back off exponentially (see
// loop.mjs) so a persistent fault -- bad credentials, Ollama down -- retries
// at a sane rate instead of spinning; any success resets that streak.
//
// ai/render_batch.py is still the manual recovery tool for re-rendering a
// jobs.json produced before this change; nothing here spawns it.
const client = new LocalClient();
const engine = new LoveEngine(client);
let bsky = null; // logged into on the first post that needs it, then reused
const loginOnce = async () => (bsky ??= await getBsky());

const t0 = Date.now();
const backoff = new Backoff();
let posted = 0;
let attempts = 0;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

for (let i = 1; ; i++) {
    console.error(`[seq] === post ${i} ===`);
    let result = null;
    for (let attempt = 1; attempt <= 2 && !result; attempt++) {
        try {
            result = await engine.generatePost((msg) => console.error(`[love] ${msg}`), { skipImage });
        } catch (err) {
            console.error(`[seq] post ${i} attempt ${attempt} failed: ${err.message}`);
        }
    }

    if (!result) {
        console.error(`[seq] post ${i} skipped after retries`);
        const wait = backoff.fail();
        attempts++;
        console.error(`[seq] retrying post ${i + 1} in ${Math.round(wait / 1000)}s (${attempts} consecutive failures)`);
        await sleep(wait);
        if (runOnce) break;
        continue;
    }

    // A post that generated is a healthy round: clear the failure streak so a
    // later failure starts from the base delay again.
    backoff.reset();
    attempts = 0;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.error(
        `[seq] #${result.transmissionNumber} ready — ${elapsed}s elapsed, ` +
        `mode ${result.mode}, ${result.callLog.length} llm calls, ` +
        `vibe "${result.vibe}"`
    );
    console.error(`[seq] ${result.text.slice(0, 80).replace(/\n/g, " ")}`);

    // generateImage already wrote a timestamped copy under output/; this is the
    // one named after the transmission, matching the existing files there.
    let imgPath = null;
    if (result.imageBlob) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
        imgPath = path.join(OUTPUT_DIR, `transmission-${result.transmissionNumber}.png`);
        fs.writeFileSync(imgPath, Buffer.from(await result.imageBlob.arrayBuffer()));
    }

    if (doPost) {
        try {
            const b = await loginOnce();
            if (imgPath) {
                await postOne(b, result.text, imgPath);
            } else {
                const res = await b.createPost(result.text);
                console.log(`posted: ${res.uri}`);
            }
            posted++;
        } catch (err) {
            // A failed post must not take the run down: the image is on disk and
            // the transmission number is spent, so carry on with the next post.
            console.error(`[seq] post #${result.transmissionNumber} failed: ${err.message}`);
        }
    }
    console.error(`[seq] post ${i} complete — ${posted} posted so far`);

    // Follow-back runs after the post lands, never before it: the SDXL render
    // and the Bluesky upload are the important work, and a scan that throws
    // must not be able to cost us a post.
    if (doPost) {
        try {
            await doFollowBack(await loginOnce(), engine, { skipImage });
        } catch (err) {
            console.error(`[follow] scan error: ${err.message}`);
        }
    }

    if (runOnce) break;
}
console.error(`[seq] done — ${posted} posted in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
