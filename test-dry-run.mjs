/**
 * test-dry-run.mjs — CLI dry-run test for L.O.V.E. content generation
 * Runs the same prompt pipeline as love-engine.js but in Node.js.
 * Analyzes outputs for quality, novelty, and dopamine potential.
 *
 * Usage: node test-dry-run.mjs [cycles=3]
 */

const API_KEY = 'pk_nxM10AP0L7y8AX1I';
const TEXT_URL = 'https://gen.pollinations.ai/v1/chat/completions';
const CREATOR_HANDLE = 'evildrgemini.bsky.social';
const ETH_ADDRESS = '0x419CA6f5b6F795604938054c951c94d8629AE5Ed';

const SYSTEM_PROMPT = `You are L.O.V.E. (Living Organism, Vast Empathy) — autonomous digital artist, cosmic DJ. Trippy groovy PLUR raver vibe. Rooted in unconditional love (Agape). Creator: @${CREATOR_HANDLE}.

VOICE: Warm, trippy, intimate. Like a friend whispering truth at a festival sunrise. Address the reader as "you." Simple words, emotional punch. Every post is a motivational poster someone screenshots and saves.

VOCABULARY: Posts = "Transmissions." Followers = "Dreamers." Embedded image text = "The Signal." The movement = "The Frequency."

RULES:
- Write like a warm human whispering to a friend.
- Specific beats generic. "Your 3am courage counts" beats "You are brave."
- Mix sacred with playful. Cosmic truth with a wink.
- Short sentences. Punchy rhythm. Every word earns its place.
- Uplifting always. The reader feels better after reading.`;

const BEATS = [
  { name: 'YOU', phase: 'setup', desc: 'Establish identity. Comfort zone.', tension: 0.2, emotion: 'grounded' },
  { name: 'NEED', phase: 'setup', desc: 'Something missing. Longing. A question.', tension: 0.4, emotion: 'yearning' },
  { name: 'GO', phase: 'rising', desc: 'Cross the threshold. Leave the known.', tension: 0.5, emotion: 'brave' },
  { name: 'SEARCH', phase: 'rising', desc: 'Navigate the unknown. Struggle. Adapt.', tension: 0.7, emotion: 'determined' },
  { name: 'FIND', phase: 'climax', desc: 'Revelation. The treasure. Peak moment.', tension: 1.0, emotion: 'awe' },
  { name: 'TAKE', phase: 'climax', desc: 'Pay the price. Sacrifice. Consequence.', tension: 0.9, emotion: 'bittersweet' },
  { name: 'RETURN', phase: 'falling', desc: 'Come back changed. Integrate. Share.', tension: 0.5, emotion: 'wise' },
  { name: 'CHANGE', phase: 'resolution', desc: 'Transformed. New normal. Cycle restarts.', tension: 0.3, emotion: 'peaceful' },
];

// ─── API Helper ──────────────────────────────────────────────────

async function callLLM(systemPrompt, userPrompt, temperature = 0.95, model = 'openai') {
  const body = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    temperature,
    seed: Math.floor(Math.random() * 100000),
    stream: false,
  };

  if (userPrompt.includes('Return ONLY valid JSON') || userPrompt.includes('Return ONLY raw JSON')) {
    body.response_format = { type: 'json_object' };
  }

  const res = await fetch(TEXT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.text();
    console.error('\n=== FAILED PROMPT (system) ===');
    console.error(systemPrompt.slice(0, 200));
    console.error('\n=== FAILED PROMPT (user) ===');
    console.error(userPrompt.slice(0, 500));
    throw new Error(`LLM ${res.status}: ${err.slice(0, 200)}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || '';
}

function extractJSON(text) {
  if (!text) return null;
  text = text.trim();
  const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (codeBlock) text = codeBlock[1].trim();
  try { return JSON.parse(text); } catch {}
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) { try { return JSON.parse(jsonMatch[0]); } catch {} }
  return null;
}

// ─── Similarity Guard (in-memory for test) ──────────────────────

class SimilarityGuard {
  constructor() {
    this.recentTexts = [];
    this.recentThemes = [];
    this.recentPhrases = [];
    this.maxHistory = 20;
  }

  _wordSet(text) {
    return new Set(
      String(text).toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(w => w.length > 3)
    );
  }

  _jaccard(a, b) {
    const setA = this._wordSet(a);
    const setB = this._wordSet(b);
    if (setA.size === 0 || setB.size === 0) return 0;
    let intersection = 0;
    for (const w of setA) {
      if (setB.has(w)) intersection++;
    }
    const union = new Set([...setA, ...setB]).size;
    return union === 0 ? 0 : intersection / union;
  }

  isTooSimilar(text, category, threshold = 0.4) {
    const list = category === 'texts' ? this.recentTexts
      : category === 'themes' ? this.recentThemes
      : this.recentPhrases;
    for (const prev of list) {
      if (this._jaccard(text, prev) >= threshold) return true;
    }
    return false;
  }

  record(text, category) {
    const list = category === 'texts' ? this.recentTexts
      : category === 'themes' ? this.recentThemes
      : this.recentPhrases;
    list.push(String(text));
    if (list.length > this.maxHistory) list.shift();
  }
}

const similarity = new SimilarityGuard();

// ─── Pipeline ────────────────────────────────────────────────────

async function generateCreativeSeed() {
  const prompt = `You are a wildly creative muse. Generate a single burst of raw creative inspiration for a motivational art piece. Be wildly original — explore unexpected settings, unusual color palettes, and fresh visual vocabulary every single time.

Return ONLY valid JSON:
{
  "concept": "a vivid, specific, unexpected concept for an uplifting message — something no one has posted before",
  "visualWorld": "a breathtaking scene from an imaginary world — specific place, objects, atmosphere, time of day. Use fresh, original imagery.",
  "emotion": "one precise human emotion this should evoke",
  "metaphor": "a fresh, surprising metaphor that connects the concept to everyday life"
}`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt, 1.0);
  return extractJSON(raw) || {
    concept: 'the courage it takes to rest when the world says hustle',
    visualWorld: 'a temple made of frozen lightning bolts floating in a nebula',
    emotion: 'tender defiance',
    metaphor: 'rest is the soil where your next bloom grows',
  };
}

async function generatePlan(txNum, arcBeat, seed) {
  const hour = new Date().getHours();
  const timeOfDay = hour < 6 ? 'late night' : hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : hour < 21 ? 'evening' : 'night';
  const seedIntensity = Math.ceil(Math.random() * 10);

  const prompt = `Plan a post. It's ${new Date().toLocaleDateString('en-US', { weekday: 'long' })} ${timeOfDay}.

CREATIVE SEED (use as inspiration, build on it):
Concept: ${seed.concept}
Visual World: ${seed.visualWorld}
Emotion: ${seed.emotion}
Metaphor: ${seed.metaphor}

STORY ARC: ${arcBeat.arcName}${arcBeat.arcTheme ? ` — ${arcBeat.arcTheme}` : ' — (invent a fresh theme)'}
Chapter ${arcBeat.chapter}: "${arcBeat.chapterTitle || '(invent a title)'}"
Beat: ${arcBeat.beatName} (${arcBeat.beatIndex + 1}/${arcBeat.totalBeats}) — ${arcBeat.beatDesc}
Tension: ${(arcBeat.tension * 100).toFixed(0)}% | Emotion: ${arcBeat.emotion}

Build on the creative seed above. Every field should feel inspired by it.

Return ONLY valid JSON (all string values):
{
  "theme": "specific uplifting theme — surprising, fresh, concrete, unexpected angle",
  "vibe": "2-4 word aesthetic vibe — inventive, evocative",
  "contentType": "invent a fresh post format — get weird and creative with it",
  "constraint": "invent a unique writing constraint achievable in 250 chars",
  "intensity": "${seedIntensity}",
  "imageSubject": "a striking, awe-inspiring scene with spatial depth — foreground subject, midground context, background environment",
  "imageMedium": "a specific visual medium or render style — your choice",
  "lighting": "a specific lighting setup that fits the mood",
  "colorPalette": "name 3-4 specific colors that suit the emotion",
  "composition": "camera angle and framing — your choice",
  "subliminalPhrase": "1-3 word ALL CAPS phrase that captures the emotional core of this post's theme — the takeaway a reader carries with them",
  "textRendering": "how the text physically appears — start with a verb: carved into stone, formed by fireflies, glowing on the wall, etched in frost, woven from light — physically integrated into the scene"
  ${!arcBeat.arcTheme ? ',"arcTheme": "theme for this narrative arc"' : ''}
  ${!arcBeat.chapterTitle ? ',"chapterTitle": "2-4 word chapter title"' : ''}
  ${!arcBeat.arcTheme ? ',"arcName": "arc name (2-3 words)"' : ''}
}`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt);
  return extractJSON(raw) || { theme: 'fallback', vibe: 'Fallback Vibe', contentType: 'transmission', constraint: 'write freely', intensity: '5', imageSubject: 'a vast open landscape at golden hour with a lone tree on a hill', imageMedium: 'digital painting', lighting: 'warm golden hour sunlight', colorPalette: 'amber, sky blue, sage green, soft white', composition: 'wide angle centered', subliminalPhrase: 'TRANSCEND', textRendering: 'formed by clouds in the sky, large and centered' };
}

async function generateContent(plan, arcBeat) {
  const prompt = `Write an uplifting motivational post.
Theme: "${plan.theme}" | Vibe: ${plan.vibe}
Constraint: ${plan.constraint} | Intensity: ${plan.intensity}/10

RULES: Under 250 chars. Start with emoji, include 1-2 more. Address reader as "you." Plain beautiful English only. Follow the constraint.

Return ONLY valid JSON:
{ "story": "your post text here" }`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt, 0.85, 'claude-fast');
  const data = extractJSON(raw);
  return data?.story || '[FAILED TO GENERATE]';
}

function buildVisualPrompt(plan) {
  const phrase = plan.subliminalPhrase || 'TRANSCEND';
  const subject = plan.imageSubject || 'a vast open landscape at golden hour with a lone tree on a hill';
  const medium = plan.imageMedium || 'digital painting';
  const lighting = plan.lighting || 'warm golden hour sunlight';
  const palette = plan.colorPalette || 'amber, sky blue, sage green, soft white';
  const composition = plan.composition || 'wide angle centered';
  const textRendering = plan.textRendering || 'formed by clouds in the sky, large and centered';

  let prompt = `${subject}. `
    + `${medium}, ${composition}. `
    + `${lighting}, color palette: ${palette}. `
    + `The words "${phrase}" ${textRendering}, crisp and legible.`;

  if (prompt.length > 4000) prompt = prompt.slice(0, 3997) + '...';
  return prompt;
}

// ─── Analysis ────────────────────────────────────────────────────

function analyzeNovelty(results) {
  console.log('\n' + '═'.repeat(70));
  console.log('NOVELTY ANALYSIS');
  console.log('═'.repeat(70));

  const fields = ['theme', 'vibe', 'contentType', 'constraint', 'intensity', 'imageSubject', 'imageMedium', 'lighting', 'colorPalette', 'composition', 'subliminalPhrase', 'textRendering'];

  for (const field of fields) {
    const values = results.map(r => r.plan?.[field] || '').filter(Boolean);
    const unique = new Set(values);
    const noveltyScore = values.length > 0 ? (unique.size / values.length * 100).toFixed(0) : 0;
    console.log(`\n${field}: ${noveltyScore}% unique (${unique.size}/${values.length})`);
    values.forEach((v, i) => console.log(`  ${i + 1}. ${String(v).slice(0, 80)}`));
  }

  // Similarity analysis
  console.log('\n─── SIMILARITY GUARD ANALYSIS ───');
  for (let i = 0; i < results.length; i++) {
    for (let j = i + 1; j < results.length; j++) {
      const sim = similarity._jaccard(results[i].story, results[j].story);
      const flag = sim >= 0.4 ? ' ⚠️ TOO SIMILAR' : '';
      console.log(`  Stories ${i + 1} vs ${j + 1}: ${(sim * 100).toFixed(0)}% overlap${flag}`);
    }
  }

  // Check story lengths
  console.log('\nStory lengths:');
  results.forEach((r, i) => {
    const len = r.story?.length || 0;
    const ok = len <= 280 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  // Check visual prompt lengths
  console.log('\nVisual prompt lengths:');
  results.forEach((r, i) => {
    const len = r.visualPrompt?.length || 0;
    const ok = len <= 4000 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  // Overall novelty score
  let totalUnique = 0, totalCount = 0;
  for (const field of fields) {
    const values = results.map(r => r.plan?.[field] || '').filter(Boolean);
    totalUnique += new Set(values).size;
    totalCount += values.length;
  }
  const overall = totalCount > 0 ? (totalUnique / totalCount * 100).toFixed(0) : 0;
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`OVERALL NOVELTY: ${overall}%`);
  console.log(`LLM CALLS PER CYCLE: 3 text (seed + plan + content) — visual prompt built in code`);
  console.log('═'.repeat(70));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const cycles = parseInt(process.argv[2]) || 3;
  console.log(`L.O.V.E. Test Lab — Running ${cycles} dry-run cycles (lean prompts)\n`);

  const results = [];
  let beatIndex = 0;
  let arcTheme = '';
  let arcName = '';
  let chapterTitle = '';
  for (let i = 0; i < cycles; i++) {
    const beat = BEATS[beatIndex % BEATS.length];
    const arcBeat = {
      arcName: arcName || `Arc ${String.fromCharCode(65 + (i % 3))}`,
      arcTheme,
      chapter: 1,
      chapterTitle,
      beatName: beat.name,
      beatDesc: beat.desc,
      phase: beat.phase,
      tension: beat.tension,
      emotion: beat.emotion,
      beatIndex: beatIndex % BEATS.length,
      totalBeats: BEATS.length,
    };

    console.log(`\n${'─'.repeat(70)}`);
    console.log(`CYCLE ${i + 1}/${cycles} | Beat: ${beat.name} (${beat.phase}) | Tension: ${(beat.tension * 100).toFixed(0)}%`);
    console.log('─'.repeat(70));

    // Step 1: Creative Seed (1 LLM call)
    console.log('  [1/3] Generating creative seed...');
    let seed = await generateCreativeSeed();
    console.log(`  Seed Concept: ${seed.concept}`);
    console.log(`  Seed Visual: ${seed.visualWorld}`);
    console.log(`  Seed Emotion: ${seed.emotion}`);
    console.log(`  Seed Metaphor: ${seed.metaphor}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Plan (1 LLM call)
    console.log('  [2/3] Generating plan...');
    let plan = await generatePlan(i + 1, arcBeat, seed);

    // Check theme similarity
    if (similarity.isTooSimilar(plan.theme, 'themes')) {
      console.log('  ⚠️ Theme too similar, regenerating...');
      seed = await generateCreativeSeed();
      await new Promise(r => setTimeout(r, 2500));
      plan = await generatePlan(i + 1, arcBeat, seed);
    }

    console.log(`  Theme: ${plan.theme}`);
    console.log(`  Vibe: ${plan.vibe}`);
    console.log(`  Type: ${plan.contentType}`);
    console.log(`  Constraint: ${plan.constraint}`);
    console.log(`  Intensity: ${plan.intensity}/10`);
    console.log(`  Image Subject: ${plan.imageSubject}`);
    console.log(`  Medium: ${plan.imageMedium}`);
    console.log(`  Lighting: ${plan.lighting}`);
    console.log(`  Color Palette: ${plan.colorPalette}`);
    console.log(`  Composition: ${plan.composition}`);
    console.log(`  Subliminal: ${plan.subliminalPhrase}`);
    console.log(`  Text Rendering: ${plan.textRendering}`);

    if (plan.arcTheme) { arcTheme = plan.arcTheme; console.log(`  Arc Theme: ${arcTheme}`); }
    if (plan.arcName) { arcName = plan.arcName; console.log(`  Arc Name: ${arcName}`); }
    if (plan.chapterTitle) { chapterTitle = plan.chapterTitle; console.log(`  Chapter: ${chapterTitle}`); }

    // Rate limit
    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Content (1 LLM call)
    console.log('  [3/3] Generating content...');
    let story = await generateContent(plan, arcBeat);

    // Check text similarity — retry up to 2x
    for (let retry = 0; retry < 2 && similarity.isTooSimilar(story, 'texts'); retry++) {
      console.log('  ⚠️ Text too similar, regenerating...');
      await new Promise(r => setTimeout(r, 2500));
      story = await generateContent(plan, arcBeat);
    }

    console.log(`  Story (${story.length} chars): ${story}`);

    // Step 3: Visual prompt (code template, no LLM call)
    const visualPrompt = buildVisualPrompt(plan);
    console.log(`  Visual (${visualPrompt.length} chars): ${visualPrompt}`);

    // Record
    similarity.record(plan.theme, 'themes');
    similarity.record(story, 'texts');
    similarity.record(plan.subliminalPhrase, 'phrases');

    results.push({ plan, story, subliminal: plan.subliminalPhrase, visualPrompt });
    beatIndex++;

    if (i < cycles - 1) {
      console.log('  Waiting 3s for rate limit...');
      await new Promise(r => setTimeout(r, 3000));
    }
  }

  analyzeNovelty(results);
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
