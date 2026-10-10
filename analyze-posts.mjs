#!/usr/bin/env node
// Measures the cadence/coherence health of the published post ring.
//
// It calls the SAME _postQuality gate the engine enforces at generation time,
// so a before/after comparison is apples to apples rather than a re-reading.
// Usage:
//   node analyze-posts.mjs [state-file]      (default: .love-state.json)

import fs from "node:fs";
import { LoveEngine } from "./public/js/love-engine.js";

const file = process.argv[2] || ".love-state.json";
const state = JSON.parse(fs.readFileSync(file, "utf8"));
const posts = JSON.parse(state.love_recent_posts || "[]");
const E = LoveEngine.prototype;

const graphemes = (t) => E._countGraphemes.call(E, t);
const sentences = (t) =>
    t.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);

let flagged = 0;
let questions = 0;
let sentsTotal = 0;
const lengths = [];
const reasonCounts = new Map();

posts.forEach((t, i) => {
    const reasons = E._postQuality.call(E, t);
    const sents = sentences(t);
    const len = graphemes(t);
    lengths.push(len);
    sentsTotal += sents.length;
    if (/\?[\s\p{Emoji}\uFE0F]*$/u.test(t.trim())) questions++;
    if (reasons.length) flagged++;
    for (const r of reasons) {
        const key = r.includes("ending is a fragment")
            ? "dangling ending (fragment)"
            : r.includes("shouted in caps")
              ? "ending shouted in caps"
              : r.includes("run-on")
                ? "run-on sentence"
                : r.includes("staccato")
                  ? "staccato"
                  : r.includes("invented compound")
                    ? "invented compound"
                    : r.includes("times in one post")
                      ? "content word used 3x+"
                      : r;
        reasonCounts.set(key, (reasonCounts.get(key) || 0) + 1);
    }
    console.log(
        `#${String(i + 1).padStart(2)} len=${String(len).padStart(3)} sent=${sents.length} ` +
            (reasons.length ? `✗ ${reasons.join(" | ")}` : "ok"),
    );
});

const n = posts.length || 1;
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const pct = (x) => `${Math.round((x / n) * 100)}%`;

console.log(
    `\nposts=${posts.length}  flagged=${flagged} (${pct(flagged)})  ` +
        `endsQuestion=${questions} (${pct(questions)})`,
);
console.log(
    `avg length=${Math.round(avg(lengths))} graphemes  avg sentences/post=${(sentsTotal / n).toFixed(1)}`,
);
console.log("reasons:");
for (const [k, v] of [...reasonCounts].sort((a, b) => b[1] - a[1]))
    console.log(`  ${String(v).padStart(2)}  ${k}`);
