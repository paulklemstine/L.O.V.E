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
const OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M";
const IMG_PY = path.join(process.env.HOME, "ai", "imgenv", "bin", "python");
const IMG_SCRIPT = path.join(process.env.HOME, "ai", "generate_image.py");
const OUTPUT_DIR = path.join(import.meta.dirname, "output");

// ─── Local client: same interface as PollinationsClient ─────────────────
class LocalClient {
    constructor() { this.callLog = []; }
    resetCallLog() { this.callLog = []; }
    getCallLog() { return this.callLog; }

    async generateText(systemPrompt, userPrompt, options = {}) {
        const { temperature = 0.85, maxRetries = 2, label = "LLM Call" } = options;
        const body = {
            model: OLLAMA_MODEL,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt },
            ],
            temperature,
            stream: false,
        };
        if (userPrompt.includes("Return ONLY valid JSON") || userPrompt.includes("Return ONLY raw JSON")) {
            body.response_format = { type: "json_object" };
        }

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const res = await fetch(`${OLLAMA_URL}/v1/chat/completions`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                if (!res.ok) throw new Error(`Ollama ${res.status}: ${(await res.text()).slice(0, 200)}`);
                const data = await res.json();
                const text = data.choices?.[0]?.message?.content || "";
                if (!text.trim()) throw new Error("Ollama returned empty response");
                this.callLog.push({ label, systemPrompt, userPrompt, response: text, model: OLLAMA_MODEL });
                return text;
            } catch (err) {
                if (attempt === maxRetries) throw err;
                await new Promise((r) => setTimeout(r, 5000 * Math.pow(2, attempt)));
            }
        }
    }

    async generateImage(prompt, options = {}) {
        const { width = 1024, height = 1024, negativePrompt = null, seed = null } = options;
        const fullPrompt = `Very erotic and sensual: ${prompt}`;

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

// ─── Main ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const doPost = args.includes("--post");
const skipImage = args.includes("--skip-image");

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
    const imgPath = path.join(OUTPUT_DIR, `transmission-${result.transmissionNumber}.png`);
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(imgPath, Buffer.from(await result.imageBlob.arrayBuffer()));
    console.log("─".repeat(60));
    console.log(`image: ${imgPath}`);
}
console.log(`llm calls: ${result.callLog.length}`);

if (doPost) {
    const handle = process.env.BLUESKY_HANDLE;
    const password = process.env.BLUESKY_APP_PASSWORD;
    if (!handle || !password) {
        console.error("missing BLUESKY_HANDLE / BLUESKY_APP_PASSWORD (set in .env)");
        process.exit(1);
    }

    // Bluesky caps blobs at 2MB; re-encode oversized PNGs as JPEG
    if (result.imageBlob && result.imageBlob.size > 1_900_000) {
        console.error("[love] image over 2MB, converting to JPEG...");
        const src = path.join(OUTPUT_DIR, "upload-src.png");
        const dst = path.join(OUTPUT_DIR, "upload.jpg");
        fs.writeFileSync(src, Buffer.from(await result.imageBlob.arrayBuffer()));
        await new Promise((resolve, reject) => {
            const proc = spawn(IMG_PY, ["-c", `
from PIL import Image
im = Image.open("${src}").convert("RGB")
im.thumbnail((1200, 1200))
im.save("${dst}", quality=88)
`]);
            proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error("jpeg conversion failed"))));
        });
        result.imageBlob = new Blob([fs.readFileSync(dst)], { type: "image/jpeg" });
    }

    console.error(`[love] posting to Bluesky as ${handle}...`);
    const bsky = new BlueskyClient();
    await bsky.login(handle, password);
    const res = await bsky.createPost(result.text, result.imageBlob);
    console.log(`posted: ${res.uri}`);
}
