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

// ─── Pipeline ────────────────────────────────────────────────────

async function generateCreativeSeed() {
  const prompt = `Generate a single burst of creative inspiration for an uplifting social media post.

Return ONLY valid JSON:
{
  "concept": "a vivid, specific message concept",
  "emotion": "one precise human emotion this should evoke",
  "metaphor": "a fresh metaphor drawn from an unexpected domain"
}`;

  const raw = await callLLM('You are a creative director.', prompt, 1.5);
  return extractJSON(raw) || {
    concept: 'transformation', emotion: 'awe', metaphor: 'metamorphosis',
  };
}

async function generatePlan(seed) {
  const hour = new Date().getHours();
  const timeOfDay = hour < 6 ? 'late night' : hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : hour < 21 ? 'evening' : 'night';
  const seedIntensity = Math.ceil(Math.random() * 10);

  const prompt = `Plan a post. It's ${new Date().toLocaleDateString('en-US', { weekday: 'long' })} ${timeOfDay}.

CREATIVE SEED:
Concept: ${seed.concept}
Emotion: ${seed.emotion}
Metaphor: ${seed.metaphor}

Build on the creative seed above. Every field should feel inspired by it.
VARIETY IS CRITICAL: Choose a world, setting, scale, and visual language that feels completely fresh. Rotate wildly between genres, cultures, eras, scales (microscopic to cosmic), and art traditions.

Return ONLY valid JSON (all string values):
{
  "theme": "an uplifting theme",
  "vibe": "2-4 word aesthetic vibe",
  "contentType": "a post format",
  "constraint": "invent a unique writing constraint achievable in 250 chars",
  "intensity": "${seedIntensity}",
  "imageMedium": "a specific art medium or visual style — rotate between wildly different traditions",
  "lighting": "a specific lighting setup — vary dramatically each time",
  "colorPalette": "3-4 specific color names — draw from different cultural and natural palettes each time",
  "composition": "camera/framing — vary between extreme close-up, aerial, panoramic, isometric, etc.",
  "subliminalPhrase": "a short ALL CAPS phrase related to the theme"
}`;

  const raw = await callLLM('You are a creative planner for uplifting social media content.', prompt);
  return extractJSON(raw) || { theme: 'fallback', vibe: 'Fallback Vibe', contentType: 'transmission', constraint: 'write freely', intensity: '5', subliminalPhrase: 'LOVE' };
}

async function generateContent(plan) {
  const prompt = `Write an uplifting motivational post.
Theme: "${plan.theme}" | Vibe: ${plan.vibe}
Constraint: ${plan.constraint} | Intensity: ${plan.intensity}/10

RULES: Under 250 chars. Start with emoji, include 1-2 more. Address reader as "you." Plain beautiful English only. Follow the constraint. Draw metaphors from unexpected domains — vary wildly between posts.

Return ONLY valid JSON:
{ "story": "your post text here" }`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt, 0.85, 'claude-fast');
  const data = extractJSON(raw);
  let story = (data?.story || '[FAILED TO GENERATE]');
  // Remove invalid @mentions entirely (no dot = not a real Bluesky handle)
  story = story.replace(/@\w+\b(?!\.\w)/g, '').replace(/\s{2,}/g, ' ').trim();
  return story;
}

async function depersonalize(text) {
  const raw = await callLLM(
    'You rewrite text as abstract scene descriptions.',
    `Rewrite this as a short scene description focusing on environments, objects, and abstract visuals. Keep the core metaphors and emotions. Return ONLY the rewritten text:\n\n"${text}"`,
    0.7
  );
  return (raw || text).replace(/^["']|["']$/g, '').trim();
}

async function buildVisualPrompt(plan, postText = '') {
  const themeText = await depersonalize(postText || plan.theme);
  const prompt = `Create an image generation prompt for an unpopulated scene inspired by this text:

"${themeText}"
Mood: ${plan.vibe}
Medium: ${plan.imageMedium || 'any'}
Lighting: ${plan.lighting || 'any'}
Color palette: ${plan.colorPalette || 'any'}
Composition: ${plan.composition || 'any'}
Motivational phrase to embed as readable text: "${plan.subliminalPhrase}"

Use the specified medium, lighting, colors, and composition. Transform the text's metaphors into a visual scene with spatial depth (foreground, midground, background). The phrase must appear as crisp, legible text integrated into the scene. Vary the text rendering method — it can be painted, carved, projected, grown, woven, pixelated, skywritten, or any other inventive method. Vary the scale from microscopic to cosmic. Choose unexpected settings across all of human experience, nature, science, and imagination.

Write a single detailed image prompt. Return ONLY the prompt text, nothing else.`;

  const raw = await callLLM(
    'You are an image prompt writer who prizes originality.',
    prompt, 1.5
  );

  let result = (raw || '').trim();
  if (result.startsWith('"') && result.endsWith('"')) result = result.slice(1, -1);
  if (result.startsWith('```')) result = result.replace(/```\w*\n?/g, '').trim();
  if (!result || result.length < 20) {
    result = `A vast open landscape at golden hour, the words "${plan.subliminalPhrase || 'TRANSCEND'}" formed by clouds`;
  }
  if (result.length > 4000) result = result.slice(0, 3997) + '...';
  return result;
}

// ─── Analysis ────────────────────────────────────────────────────

function analyzeNovelty(results) {
  console.log('\n' + '═'.repeat(70));
  console.log('NOVELTY ANALYSIS');
  console.log('═'.repeat(70));

  const fields = ['theme', 'vibe', 'contentType', 'constraint', 'intensity', 'imagePrompt', 'subliminalPhrase'];

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
  for (let i = 0; i < cycles; i++) {
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`CYCLE ${i + 1}/${cycles}`);
    console.log('─'.repeat(70));

    // Step 1: Creative Seed (1 LLM call)
    console.log('  [1/3] Generating creative seed...');
    const seed = await generateCreativeSeed();
    console.log(`  Seed Concept: ${seed.concept}`);
    console.log(`  Seed Emotion: ${seed.emotion}`);
    console.log(`  Seed Metaphor: ${seed.metaphor}`);

    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Plan (1 LLM call)
    console.log('  [2/3] Generating plan...');
    const plan = await generatePlan(seed);

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

    // Rate limit
    await new Promise(r => setTimeout(r, 2500));

    // Step 3: Content (1 LLM call)
    console.log('  [3/3] Generating content...');
    const story = await generateContent(plan);

    console.log(`  Story (${story.length} chars): ${story}`);

    // Step 4: Visual prompt (LLM call)
    const visualPrompt = await buildVisualPrompt(plan, story);
    console.log(`  Visual (${visualPrompt.length} chars): ${visualPrompt}`);

    results.push({ plan, story, subliminal: plan.subliminalPhrase, visualPrompt });

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
