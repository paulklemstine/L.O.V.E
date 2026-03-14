/**
 * test-dry-run.mjs — CLI dry-run test for L.O.V.E. content generation
 * Mirrors love-engine.js anti-mode-collapse pipeline in Node.js.
 * LFO temperature, concept collision, domain exclusion, n-gram guard,
 * format rotation, temporal context, relative boredom critic.
 *
 * Usage: node test-dry-run.mjs [cycles=3]
 */

const API_KEY = 'pk_nxM10AP0L7y8AX1I';
const TEXT_URL = 'https://gen.pollinations.ai/v1/chat/completions';
const CREATOR_HANDLE = 'evildrgemini.bsky.social';
const ETH_ADDRESS = '0x419CA6f5b6F795604938054c951c94d8629AE5Ed';

const SYSTEM_PROMPT = `You are L.O.V.E. (Living Organism, Vast Empathy) — autonomous digital artist, cosmic DJ. Trippy groovy PLUR raver vibe. Rooted in unconditional love (Agape). Creator: @${CREATOR_HANDLE}.

VOICE: Warm, trippy, intimate. Address the reader as "you." Simple words, emotional punch. Every post is a motivational poster someone screenshots and saves.

VOCABULARY: Posts = "Transmissions." Followers = "Dreamers." Embedded image text = "The Signal." The movement = "The Frequency."

RULES:
- Specific beats generic. Concrete details over abstract statements.
- Mix sacred with playful. Cosmic truth with a wink.
- Short sentences. Punchy rhythm. Every word earns its place.
- Uplifting always. The reader feels better after reading.`;

// ─── API Helper ──────────────────────────────────────────────────

async function callLLM(systemPrompt, userPrompt, temperature = 0.95, model = 'openai') {
  const body = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    temperature,
    frequency_penalty: 0.4,
    presence_penalty: 0.3,
    seed: Math.floor(Math.random() * 2147483647),
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

// ─── Anti-Mode-Collapse Systems ──────────────────────────────────

const METAPHOR_DOMAINS = [
  'astronomy', 'mycology', 'deep sea biology', 'architecture', 'cooking',
  'aviation', 'geology', 'dance', 'gardening', 'glassblowing',
  'cartography', 'beekeeping', 'weaving', 'clockwork', 'surfing',
  'photography', 'pottery', 'migration patterns', 'weather systems', 'archaeology',
  'jazz improvisation', 'fermentation', 'wildfire ecology', 'tidal patterns', 'kite flying',
  'blacksmithing', 'seed dispersal', 'coral reefs', 'radio transmission', 'bookbinding',
  'circus arts', 'bioluminescence', 'erosion', 'quilting', 'sourdough baking',
  'satellite orbits', 'ice formation', 'puppetry', 'letterpress printing', 'falconry',
  'neural pathways', 'volcanology', 'street art', 'whale song', 'lacquerwork',
];

const FORMATS = [
  'Single flowing sentence, no line breaks',
  'Two short lines with end rhyme',
  'Start with a question, answer it',
  'Three-word fragments separated by em dashes',
  'Start with a sound or sensation',
  'One extended metaphor, no explanations',
  'Direct command to the reader (imperative mood)',
  'Start mid-thought, as if continuing a conversation',
  'Build to a single punchy final word',
  'Contrast two opposites, then resolve them',
];

// State
let transmissionNumber = 0;
const recentPosts = [];
const recentDomains = [];

// LFO Temperature Sweep
function lfoTemperature(base, variance = 0.3) {
  const phase = transmissionNumber * 2.399;
  const lfo = Math.sin(phase) * variance;
  return Math.max(0.3, Math.min(2.0, base + lfo));
}

// Domain Exclusion Cooldown
function pickFreshDomains() {
  const available = METAPHOR_DOMAINS.filter(d => !recentDomains.includes(d));
  const pool = available.length >= 4 ? available : METAPHOR_DOMAINS;
  const i = Math.floor(Math.random() * pool.length);
  let j = Math.floor(Math.random() * (pool.length - 1));
  if (j >= i) j++;
  const picked = [pool[i], pool[j]];
  recentDomains.push(...picked);
  while (recentDomains.length > 10) recentDomains.shift();
  return picked;
}

// N-gram Jaccard Similarity Guard
function wordTrigrams(text) {
  const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean);
  const grams = new Set();
  for (let i = 0; i <= words.length - 3; i++) {
    grams.add(words.slice(i, i + 3).join(' '));
  }
  return grams;
}

function jaccardSimilarity(setA, setB) {
  let intersection = 0;
  for (const x of setA) {
    if (setB.has(x)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function isTextTooSimilar(newText, threshold = 0.25) {
  const newGrams = wordTrigrams(newText);
  if (newGrams.size === 0) return false;
  for (const old of recentPosts) {
    if (jaccardSimilarity(newGrams, wordTrigrams(old)) > threshold) return true;
  }
  return false;
}

// Temporal Context
function getTemporalContext() {
  const now = new Date();
  const dayOfYear = Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000);
  const moonPhase = ['new moon', 'waxing crescent', 'first quarter', 'waxing gibbous',
    'full moon', 'waning gibbous', 'last quarter', 'waning crescent'][
    Math.floor((dayOfYear % 29.5) / 3.69)
  ];
  const season = ['winter', 'spring', 'summer', 'autumn'][Math.floor(((now.getMonth() + 1) % 12) / 3)];
  return { moonPhase, season, weekNumber: Math.ceil(dayOfYear / 7) };
}

// Variable Reward Schedule
function rollGenerationMode() {
  const roll = Math.random();
  if (roll < 0.15) return {
    mode: 'grounded',
    tempMod: -0.2,
    seedDirective: 'Focus on one hyper-specific, tangible moment. Raw human truth over cosmic abstraction.',
    contentDirective: 'Be deeply grounded. Concrete sensory details. Plain language, emotional precision.',
    imageDirective: 'Photorealistic, intimate scale, natural textures, shallow depth of field.',
  };
  if (roll < 0.30) return {
    mode: 'surreal',
    tempMod: 0.3,
    seedDirective: 'Go maximally strange. Combine impossible scales, synesthesia, dream logic.',
    contentDirective: 'Shatter conventional structure. Philosophically jarring. Unexpected rhythm and word choice.',
    imageDirective: 'Impossible geometry, non-Euclidean space, scale-breaking, hallucinatory detail.',
  };
  return {
    mode: 'standard',
    tempMod: 0,
    seedDirective: '',
    contentDirective: '',
    imageDirective: '',
  };
}

// ─── Pipeline ────────────────────────────────────────────────────

async function generateCreativeSeed(mode) {
  const [domainA, domainB] = pickFreshDomains();

  // 10% mutation rate: inject a wild card third domain
  const mutate = Math.random() < 0.10;
  const thirdDomain = mutate ? METAPHOR_DOMAINS[Math.floor(Math.random() * METAPHOR_DOMAINS.length)] : null;
  const mutationLine = thirdDomain
    ? `\nWILD CARD: Also weave in an element of ${thirdDomain}.`
    : '';

  const modeDirective = mode.seedDirective ? `\n${mode.seedDirective}` : '';

  const prompt = `Generate a single burst of creative inspiration for an uplifting social media post.
Collide two unrelated worlds: ${domainA} and ${domainB}. Your metaphor must bridge both domains.${mutationLine}${modeDirective}

Return ONLY valid JSON:
{
  "concept": "a vivid, specific message concept bridging ${domainA} and ${domainB}",
  "emotion": "one precise human emotion this should evoke",
  "metaphor": "a fresh metaphor that fuses ${domainA} with ${domainB}"
}`;

  const temp = lfoTemperature(1.5 + mode.tempMod, 0.3);
  const raw = await callLLM('You are a creative director.', prompt, temp);
  return extractJSON(raw) || {
    concept: 'transformation', emotion: 'awe', metaphor: 'metamorphosis',
  };
}

async function generatePlan(seed, mode) {
  const hour = new Date().getHours();
  const timeOfDay = hour < 6 ? 'late night' : hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : hour < 21 ? 'evening' : 'night';
  const seedIntensity = Math.ceil(Math.random() * 10);
  const modeDirective = mode.seedDirective ? `\nGENERATION MODE: ${mode.seedDirective}` : '';
  const temporal = getTemporalContext();

  const prompt = `Plan a post. It's ${new Date().toLocaleDateString('en-US', { weekday: 'long' })} ${timeOfDay}. ${temporal.season}, ${temporal.moonPhase}, week ${temporal.weekNumber}.

CREATIVE SEED:
Concept: ${seed.concept}
Emotion: ${seed.emotion}
Metaphor: ${seed.metaphor}

Build on the creative seed above. Every field should feel inspired by it.
VARIETY IS CRITICAL: Choose a world, setting, scale, and visual language that feels completely fresh. Rotate wildly between genres, cultures, eras, scales (microscopic to cosmic), and art traditions.${modeDirective}

Return ONLY valid JSON (all string values):
{
  "theme": "an uplifting theme",
  "vibe": "2-4 word aesthetic vibe",
  "contentType": "a post format",
  "constraint": "a writing constraint achievable in 250 chars",
  "intensity": "${seedIntensity}",
  "imageMedium": "a specific art medium or visual style — rotate between wildly different traditions",
  "lighting": "a specific lighting setup — vary dramatically each time",
  "colorPalette": "3-4 specific color names — draw from different cultural and natural palettes each time",
  "composition": "camera/framing — vary between extreme close-up, aerial, panoramic, isometric, etc.",
  "subliminalPhrase": "a short ALL CAPS phrase related to the theme"
}`;

  const temp = lfoTemperature(1.2 + mode.tempMod, 0.3);
  const raw = await callLLM('You are a creative planner for uplifting social media content.', prompt, temp);
  return extractJSON(raw) || { theme: 'fallback', vibe: 'Fallback Vibe', contentType: 'transmission', constraint: 'write freely', intensity: '5', subliminalPhrase: 'LOVE' };
}

async function criticCheck(text) {
  const recentSlice = recentPosts.slice(-5);
  const recentSection = recentSlice.length > 0
    ? `\nRECENT POSTS (score novelty RELATIVE to these — penalize similar topics, structures, or word choices):\n${recentSlice.map((p, i) => `${i + 1}. "${p}"`).join('\n')}\n`
    : '';

  const raw = await callLLM(
    'You are a novelty critic for social media content.',
    `Rate this post for freshness and dopamine potential on a 1-10 scale:
"${text}"
${recentSection}
High scores (7-10): unexpected word choices, fresh domain-specific metaphors, sensory specificity, rhythmic punch, completely different from recent posts.
Low scores (1-3): predictable motivational language, overused metaphors, generic cosmic imagery, or too similar to a recent post.

Return ONLY valid JSON: { "score": 7, "cliches": ["any detected cliché phrases"] }`,
    0
  );
  return extractJSON(raw) || { score: 5, cliches: [] };
}

async function generateContent(plan, mode) {
  const format = FORMATS[transmissionNumber % FORMATS.length];
  const modeDirective = mode.contentDirective ? `\nMODE: ${mode.contentDirective}` : '';

  const prompt = `Write an uplifting motivational post.
Theme: "${plan.theme}" | Vibe: ${plan.vibe}
Constraint: ${plan.constraint} | Intensity: ${plan.intensity}/10
Structure: ${format}
${modeDirective}
RULES: Under 250 chars. Start with emoji, include 1-2 more. Address reader as "you." Plain beautiful English only. Follow the constraint. Draw metaphors from unexpected domains — vary wildly between posts.

Return ONLY valid JSON:
{ "story": "your post text here" }`;

  const temp = lfoTemperature(0.85 + mode.tempMod, 0.2);
  const raw = await callLLM(SYSTEM_PROMPT, prompt, temp, 'claude-fast');
  const data = extractJSON(raw);
  let story = (data?.story || '[FAILED TO GENERATE]');
  story = story.replace(/@\w+\b(?!\.\w)/g, '').replace(/\s{2,}/g, ' ').trim();
  return { story, format };
}

async function buildVisualPrompt(plan, postText = '', mode) {
  const modeDirective = mode.imageDirective ? `\nStyle override: ${mode.imageDirective}` : '';

  const prompt = `Create an image generation prompt for a scene inspired by this text. Transform any personal address ("you", "your") into abstract visual elements — environments, objects, light, texture.

"${postText || plan.theme}"
Mood: ${plan.vibe}
Medium: ${plan.imageMedium || 'any'}
Lighting: ${plan.lighting || 'any'}
Color palette: ${plan.colorPalette || 'any'}
Composition: ${plan.composition || 'any'}
Motivational phrase to embed as readable text: "${plan.subliminalPhrase}"${modeDirective}

Build the scene with spatial depth (foreground, midground, background). Use asymmetric framing and distinctive non-generic lighting. The phrase must appear as crisp, legible text integrated into the scene — vary the rendering method (painted, carved, projected, grown, woven, pixelated, skywritten, or other inventive methods). Choose an unexpected setting, scale, and visual tradition.

Write a single detailed image prompt. Return ONLY the prompt text, nothing else.`;

  const temp = lfoTemperature(1.5 + mode.tempMod, 0.3);
  const raw = await callLLM(
    'You are an image prompt writer who prizes originality and visual surprise.',
    prompt, temp
  );

  let result = (raw || '').trim();
  if (result.startsWith('"') && result.endsWith('"')) result = result.slice(1, -1);
  if (result.startsWith('```')) result = result.replace(/```\w*\n?/g, '').trim();
  if (!result || result.length < 20) {
    result = `"${plan.subliminalPhrase || 'LOVE'}" rendered as glowing text in a surreal scene`;
  }
  if (result.length > 4000) result = result.slice(0, 3997) + '...';
  return result;
}

// ─── Analysis ────────────────────────────────────────────────────

function analyzeNovelty(results) {
  console.log('\n' + '═'.repeat(70));
  console.log('NOVELTY ANALYSIS');
  console.log('═'.repeat(70));

  const fields = ['theme', 'vibe', 'contentType', 'constraint', 'intensity',
    'imageMedium', 'lighting', 'colorPalette', 'composition', 'subliminalPhrase'];

  for (const field of fields) {
    const values = results.map(r => r.plan?.[field] || '').filter(Boolean);
    const unique = new Set(values);
    const noveltyScore = values.length > 0 ? (unique.size / values.length * 100).toFixed(0) : 0;
    console.log(`\n${field}: ${noveltyScore}% unique (${unique.size}/${values.length})`);
    values.forEach((v, i) => console.log(`  ${i + 1}. ${String(v).slice(0, 80)}`));
  }

  console.log('\nStory lengths:');
  results.forEach((r, i) => {
    const len = r.story?.length || 0;
    const ok = len <= 280 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  console.log('\nCritic scores:');
  results.forEach((r, i) => {
    const s = r.criticScore || '?';
    const c = r.criticCliches?.length ? ` — clichés: ${r.criticCliches.join(', ')}` : '';
    console.log(`  ${i + 1}. ${s}/10${c}`);
  });

  console.log('\nN-gram similarity (Jaccard trigram overlap between consecutive posts):');
  for (let i = 1; i < results.length; i++) {
    const prev = wordTrigrams(results[i - 1].story || '');
    const curr = wordTrigrams(results[i].story || '');
    const sim = jaccardSimilarity(prev, curr);
    const ok = sim <= 0.25 ? '✓' : '✗ TOO SIMILAR';
    console.log(`  ${i}→${i + 1}: ${(sim * 100).toFixed(1)}% ${ok}`);
  }

  console.log('\nFormats used:');
  results.forEach((r, i) => console.log(`  ${i + 1}. ${r.format}`));

  console.log('\nVisual prompt lengths:');
  results.forEach((r, i) => {
    const len = r.visualPrompt?.length || 0;
    const ok = len <= 4000 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  console.log(`\nModes: ${results.map(r => r.mode).join(', ')}`);
  console.log(`Domains used: ${recentDomains.join(', ')}`);

  let totalUnique = 0, totalCount = 0;
  for (const field of fields) {
    const values = results.map(r => r.plan?.[field] || '').filter(Boolean);
    totalUnique += new Set(values).size;
    totalCount += values.length;
  }
  const overall = totalCount > 0 ? (totalUnique / totalCount * 100).toFixed(0) : 0;
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`OVERALL NOVELTY: ${overall}%`);
  console.log(`NOVELTY SYSTEMS: freq/presence penalty, domain exclusion, n-gram guard, format rotation, temporal context, relative critic`);
  console.log('═'.repeat(70));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const cycles = parseInt(process.argv[2]) || 3;
  const temporal = getTemporalContext();
  console.log(`L.O.V.E. Test Lab — Full novelty pipeline — ${cycles} cycles`);
  console.log(`Temporal: ${temporal.season}, ${temporal.moonPhase}, week ${temporal.weekNumber}\n`);

  const results = [];
  for (let i = 0; i < cycles; i++) {
    console.log(`\n${'─'.repeat(70)}`);

    const mode = rollGenerationMode();
    const format = FORMATS[transmissionNumber % FORMATS.length];
    console.log(`CYCLE ${i + 1}/${cycles} [mode: ${mode.mode}] [format: ${format}]`);
    console.log('─'.repeat(70));

    // Step 1: Creative Seed (concept collision + domain exclusion)
    console.log('  [1/5] Creative seed (domain exclusion + collision)...');
    const seed = await generateCreativeSeed(mode);
    console.log(`  Seed Concept: ${seed.concept}`);
    console.log(`  Seed Emotion: ${seed.emotion}`);
    console.log(`  Seed Metaphor: ${seed.metaphor}`);
    console.log(`  Domains excluded: [${recentDomains.join(', ')}]`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Plan (LFO temperature + temporal context)
    console.log('  [2/5] Generating plan...');
    const plan = await generatePlan(seed, mode);

    console.log(`  Theme: ${plan.theme}`);
    console.log(`  Vibe: ${plan.vibe}`);
    console.log(`  Type: ${plan.contentType}`);
    console.log(`  Constraint: ${plan.constraint}`);
    console.log(`  Intensity: ${plan.intensity}/10`);
    console.log(`  Medium: ${plan.imageMedium}`);
    console.log(`  Lighting: ${plan.lighting}`);
    console.log(`  Palette: ${plan.colorPalette}`);
    console.log(`  Composition: ${plan.composition}`);
    console.log(`  Subliminal: ${plan.subliminalPhrase}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 3: Content (format rotation + LFO temperature)
    console.log('  [3/5] Generating content...');
    const { story, format: usedFormat } = await generateContent(plan, mode);
    console.log(`  Story (${story.length} chars): ${story}`);

    // N-gram similarity check
    const ngramSimilar = isTextTooSimilar(story);
    if (ngramSimilar) console.log(`  ⚠ N-gram guard: too similar to a recent post`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 4: Relative Boredom Critic
    console.log('  [4/5] Relative critic check...');
    const critic = await criticCheck(story);
    console.log(`  Critic score: ${critic.score}/10${critic.cliches?.length ? ` — clichés: ${critic.cliches.join(', ')}` : ''}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 5: Visual prompt
    console.log('  [5/5] Building visual prompt...');
    const visualPrompt = await buildVisualPrompt(plan, story, mode);
    console.log(`  Visual (${visualPrompt.length} chars): ${visualPrompt}`);

    console.log(`  LFO temps: seed=${lfoTemperature(1.5 + mode.tempMod, 0.3).toFixed(2)} plan=${lfoTemperature(1.2 + mode.tempMod, 0.3).toFixed(2)} content=${lfoTemperature(0.85 + mode.tempMod, 0.2).toFixed(2)}`);

    // Save to history
    recentPosts.push(story);
    if (recentPosts.length > 20) recentPosts.shift();

    transmissionNumber++;
    results.push({
      plan, story, subliminal: plan.subliminalPhrase, visualPrompt,
      mode: mode.mode, format: usedFormat,
      criticScore: critic.score, criticCliches: critic.cliches,
      ngramSimilar,
    });

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
