/**
 * test-dry-run.mjs — CLI dry-run test for L.O.V.E. content generation
 * Mirrors love-engine.js anti-mode-collapse pipeline in Node.js.
 * LFO temperature sweep, concept collisions, boredom critic,
 * variable reward schedule, mutation injection.
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

// LFO Temperature Sweep — golden angle oscillation avoids repeating patterns
let transmissionNumber = 0;
function lfoTemperature(base, variance = 0.3) {
  const phase = transmissionNumber * 2.399;
  const lfo = Math.sin(phase) * variance;
  return Math.max(0.3, Math.min(2.0, base + lfo));
}

// Variable Reward Schedule — dopamine from reward prediction error
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
  // Concept Collision: pick 2 unrelated domains and force bridging
  const i = Math.floor(Math.random() * METAPHOR_DOMAINS.length);
  let j = Math.floor(Math.random() * (METAPHOR_DOMAINS.length - 1));
  if (j >= i) j++;
  const domainA = METAPHOR_DOMAINS[i];
  const domainB = METAPHOR_DOMAINS[j];

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

  const prompt = `Plan a post. It's ${new Date().toLocaleDateString('en-US', { weekday: 'long' })} ${timeOfDay}.

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
  const raw = await callLLM(
    'You are a novelty critic for social media content.',
    `Rate this post for freshness and dopamine potential on a 1-10 scale:
"${text}"

High scores (7-10): unexpected word choices, fresh domain-specific metaphors, sensory specificity, rhythmic punch, makes you stop scrolling.
Low scores (1-3): predictable motivational language, overused metaphors, generic cosmic imagery, safe and forgettable.

Return ONLY valid JSON: { "score": 7, "cliches": ["any detected cliché phrases"] }`,
    0
  );
  return extractJSON(raw) || { score: 5, cliches: [] };
}

async function generateContent(plan, mode) {
  const modeDirective = mode.contentDirective ? `\nMODE: ${mode.contentDirective}` : '';

  const prompt = `Write an uplifting motivational post.
Theme: "${plan.theme}" | Vibe: ${plan.vibe}
Constraint: ${plan.constraint} | Intensity: ${plan.intensity}/10
${modeDirective}
RULES: Under 250 chars. Start with emoji, include 1-2 more. Address reader as "you." Plain beautiful English only. Follow the constraint. Draw metaphors from unexpected domains — vary wildly between posts.

Return ONLY valid JSON:
{ "story": "your post text here" }`;

  const temp = lfoTemperature(0.85 + mode.tempMod, 0.2);
  const raw = await callLLM(SYSTEM_PROMPT, prompt, temp, 'claude-fast');
  const data = extractJSON(raw);
  let story = (data?.story || '[FAILED TO GENERATE]');
  story = story.replace(/@\w+\b(?!\.\w)/g, '').replace(/\s{2,}/g, ' ').trim();
  return story;
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

  // Check story lengths
  console.log('\nStory lengths:');
  results.forEach((r, i) => {
    const len = r.story?.length || 0;
    const ok = len <= 280 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  // Critic scores
  console.log('\nCritic scores:');
  results.forEach((r, i) => {
    const s = r.criticScore || '?';
    const c = r.criticCliches?.length ? ` — clichés: ${r.criticCliches.join(', ')}` : '';
    console.log(`  ${i + 1}. ${s}/10${c}`);
  });

  // Visual prompt lengths
  console.log('\nVisual prompt lengths:');
  results.forEach((r, i) => {
    const len = r.visualPrompt?.length || 0;
    const ok = len <= 4000 ? '✓' : '✗ OVER';
    console.log(`  ${i + 1}. ${len} chars ${ok}`);
  });

  // Mode distribution
  const modes = results.map(r => r.mode);
  console.log(`\nModes: ${modes.join(', ')}`);

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
  console.log(`LLM CALLS PER CYCLE: seed + plan + content + critic + visual = 5 text`);
  console.log('═'.repeat(70));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const cycles = parseInt(process.argv[2]) || 3;
  console.log(`L.O.V.E. Test Lab — Anti-mode-collapse pipeline — ${cycles} cycles\n`);

  const results = [];
  for (let i = 0; i < cycles; i++) {
    console.log(`\n${'─'.repeat(70)}`);

    // Roll generation mode (variable reward schedule)
    const mode = rollGenerationMode();
    console.log(`CYCLE ${i + 1}/${cycles} [mode: ${mode.mode}]`);
    console.log('─'.repeat(70));

    // Step 1: Creative Seed (concept collision)
    console.log('  [1/5] Generating creative seed (concept collision)...');
    const seed = await generateCreativeSeed(mode);
    console.log(`  Seed Concept: ${seed.concept}`);
    console.log(`  Seed Emotion: ${seed.emotion}`);
    console.log(`  Seed Metaphor: ${seed.metaphor}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Plan (LFO temperature)
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

    // Step 3: Content (LFO temperature)
    console.log('  [3/5] Generating content...');
    const story = await generateContent(plan, mode);
    console.log(`  Story (${story.length} chars): ${story}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 4: Boredom Critic
    console.log('  [4/5] Critic check...');
    const critic = await criticCheck(story);
    console.log(`  Critic score: ${critic.score}/10${critic.cliches?.length ? ` — clichés: ${critic.cliches.join(', ')}` : ''}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 5: Visual prompt (depersonalize folded in)
    console.log('  [5/5] Building visual prompt...');
    const visualPrompt = await buildVisualPrompt(plan, story, mode);
    console.log(`  Visual (${visualPrompt.length} chars): ${visualPrompt}`);

    console.log(`  LFO temp at cycle ${i}: seed=${lfoTemperature(1.5 + mode.tempMod, 0.3).toFixed(2)} plan=${lfoTemperature(1.2 + mode.tempMod, 0.3).toFixed(2)} content=${lfoTemperature(0.85 + mode.tempMod, 0.2).toFixed(2)}`);

    transmissionNumber++;
    results.push({
      plan, story, subliminal: plan.subliminalPhrase, visualPrompt,
      mode: mode.mode,
      criticScore: critic.score, criticCliches: critic.cliches,
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
