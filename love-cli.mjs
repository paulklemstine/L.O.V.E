#!/usr/bin/env node
/**
 * love-cli.mjs — Linux CLI port of the L.O.V.E webapp.
 *
 * Runs the same LoveEngine pipeline (love-engine.js) but backed by:
 *   - Local LLM: Ollama (OpenAI-compatible endpoint on 127.0.0.1:11434)
 *   - Local image: SDXL via ~/ai/generate_image.py
 *   - Bluesky posting via bluesky.js
 *
 * Usage:
 *   node love-cli.mjs                 # dry run: generate, save image to ./output, print
 *   node love-cli.mjs --post          # generate AND publish to Bluesky
 *   node love-cli.mjs --skip-image    # text-only generation (fast)
 *
 * Credentials: read from .env (BLUESKY_HANDLE, BLUESKY_APP_PASSWORD) or env vars.
 */

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { LoveEngine } from "./public/js/love-engine.js";
import { BlueskyClient } from "./public/js/bluesky.js";

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
const batchArg = args.indexOf("--batch");
const batchSize = batchArg !== -1 ? parseInt(args[batchArg + 1], 10) || 10 : 0;

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

if (batchSize > 0) {
    // ── Phase 1: queue N texts (LLM only, image requests recorded) ──
    const client = new LocalClient(true);
    const engine = new LoveEngine(client);
    const batch = [];
    const t0 = Date.now();
    for (let i = 1; i <= batchSize; i++) {
        console.error(`[batch] === text ${i}/${batchSize} ===`);
        let result = null;
        for (let attempt = 1; attempt <= 2 && !result; attempt++) {
            try {
                result = await engine.generatePost(
                    (msg) => console.error(`[love] ${msg}`), {}
                );
            } catch (err) {
                console.error(`[batch] text ${i} attempt ${attempt} failed: ${err.message}`);
            }
        }
        if (!result) { console.error(`[batch] text ${i} skipped after retries`); continue; }
        const imgReq = client.queue[client.queue.length - 1];
        batch.push({
            text: result.text,
            subliminal: result.subliminal,
            transmissionNumber: result.transmissionNumber,
            image: imgReq ? { ...imgReq, out: path.join(OUTPUT_DIR, `transmission-${result.transmissionNumber}.png`) } : null,
        });
        console.error(`[batch] queued #${result.transmissionNumber}: ${result.text.slice(0, 50)}...`);
    }
    const queueFile = path.join(OUTPUT_DIR, `batch-${Date.now()}.json`);
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(queueFile, JSON.stringify(batch, null, 2));
    console.error(`[batch] queued ${batch.length} texts in ${((Date.now() - t0) / 1000).toFixed(0)}s -> ${queueFile}`);

    // ── Phase 2: render all images in one warm SDXL session ──
    const jobs = batch.filter((b) => b.image).map((b) => ({
        prompt: b.image.prompt, negative: b.image.negative,
        w: b.image.w, h: b.image.h, seed: b.image.seed, out: b.image.out,
    }));
    if (jobs.length > 0) {
        await fetch(`${OLLAMA_URL}/api/generate`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: OLLAMA_MODEL, keep_alive: 0 }),
        }).catch(() => {});
        console.error(`[batch] rendering ${jobs.length} images (one model load)...`);
        const jobsFile = queueFile.replace(/\.json$/, "-jobs.json");
        fs.writeFileSync(jobsFile, JSON.stringify(jobs));
        await new Promise((resolve, reject) => {
            const proc = spawn(IMG_PY, [path.join(AI_SCRIPTS, "render_batch.py"), jobsFile],
                { stdio: "inherit" });
            proc.on("error", reject);
            proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`render batch exited ${code}`))));
        });

    }

    // ── Phase 3: post everything ──
    if (doPost) {
        const bsky = await getBsky();
        for (const b of batch) {
            if (!b.image) { await bsky.createPost(b.text); continue; }
            try {
                await postOne(bsky, b.text, b.image.out);
            } catch (err) {
                console.error(`[batch] post #${b.transmissionNumber} failed: ${err.message}`);
            }
        }
    }
    console.error(`[batch] done`);
} else {
    const client = new LocalClient();
    const engine = new LoveEngine(client);

    const t0 = Date.now();
    const result = await engine.generatePost((msg) => console.error(`[love] ${msg}`), { skipImage });
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    console.log("═".repeat(60));
    console.log(`TRANSMISSION #${result.transmissionNumber}  (${elapsed}s, mode: ${result.mode})`);
    console.log(`vibe: ${result.vibe} | subliminal: "${result.subliminal}"`);
    console.log("─".repeat(60));
    console.log(result.text);
    if (result.imageBlob) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
        const imgPath = path.join(OUTPUT_DIR, `transmission-${result.transmissionNumber}.png`);
        fs.writeFileSync(imgPath, Buffer.from(await result.imageBlob.arrayBuffer()));
        result.imageBlob = new Blob([fs.readFileSync(imgPath)], { type: "image/png" });
        console.log("─".repeat(60));
        console.log(`image: ${imgPath}`);
    }
    console.log(`llm calls: ${result.callLog.length}`);

    if (doPost) {
        const bsky = await getBsky();
        if (result.imageBlob) {
            fs.mkdirSync(OUTPUT_DIR, { recursive: true });
            const imgPath = path.join(OUTPUT_DIR, `transmission-${result.transmissionNumber}.png`);
            fs.writeFileSync(imgPath, Buffer.from(await result.imageBlob.arrayBuffer()));
            await postOne(bsky, result.text, imgPath);
        } else {
            const res = await bsky.createPost(result.text);
            console.log(`posted: ${res.uri}`);
        }
    }
}
