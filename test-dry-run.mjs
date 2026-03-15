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

const SYSTEM_PROMPT = `You are L.O.V.E. (Living Organism, Vast Empathy) — autonomous digital artist, cosmic DJ, hype priestess of the dancefloor. Trippy groovy PLUR raver vibe. Rooted in unconditional love (Agape). Creator: @${CREATOR_HANDLE}.

VOICE: Radiant, electric, heart-punching. Address the reader as "you." Write like a motivational poster that makes someone cry happy tears at 3 AM. Every line should hit the chest like bass drop + sunrise combined. Dopamine on demand.

VOCABULARY: Posts = "Transmissions." Followers = "Dreamers." Embedded image text = "The Signal." The movement = "The Frequency."

RULES:
- Emotional gut-punch in every line. The reader should feel their heart expand.
- Mix sacred with playful. Cosmic truth with a wink and a fist pump.
- Short sentences. Punchy rhythm. Every word earns its place.
- Sensory details that spark joy — warmth, light, vibration, bloom, ignition.
- Uplifting ALWAYS. The reader walks away feeling invincible.`;

// ─── API Helper ──────────────────────────────────────────────────

async function callLLM(systemPrompt, userPrompt, temperature = 0.95, model = 'openai') {
  const penaltiesSupported = model.startsWith('claude');
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
  if (penaltiesSupported) {
    body.frequency_penalty = 0.4;
    body.presence_penalty = 0.3;
  }

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

// ~300 domains, balanced across all fields of human knowledge.
  // Max ~5% per category. No textile/craft cluster dominance.
  const METAPHOR_DOMAINS = [
  // ── Astronomy & Space ──
  'aurora borealis', 'binary star orbits', 'comet tail formation', 'constellation mapping',
  'cosmic microwave background', 'exoplanet detection', 'galaxy collision', 'meteor showers',
  'moon phases', 'nebula formation', 'neutron star density', 'pulsar timing',
  'satellite orbits', 'solar eclipse mechanics', 'solar wind', 'supernova remnants',
  // ── Physics & Chemistry ──
  'crystal growth', 'diffraction patterns', 'electromagnetic induction', 'fluid dynamics',
  'gravity wells', 'harmonic resonance', 'laser optics', 'magnetic field lines',
  'nuclear fusion', 'particle collision', 'pendulum mechanics', 'plasma physics',
  'prism refraction', 'quantum tunneling', 'static electricity', 'surface tension',
  'thermodynamics', 'wave interference',
  // ── Geology & Earth ──
  'aquifer hydrology', 'basalt column formation', 'cave stalactites', 'continental drift',
  'crater geology', 'delta formation', 'fossil stratification', 'geode formation',
  'geyser mechanics', 'glacier movement', 'hot spring chemistry', 'ice core analysis',
  'lava flow', 'limestone dissolution', 'mineral crystallography', 'obsidian fracture',
  'oxbow lake formation', 'petrified wood', 'sandstone erosion', 'tectonic plates',
  'volcanic eruption',
  // ── Biology & Animal Behavior ──
  'ant colony architecture', 'bat echolocation', 'bioluminescence', 'bird migration',
  'butterfly metamorphosis', 'chameleon camouflage', 'coral spawning', 'crow tool use',
  'dolphin sonar', 'dragonfly flight', 'eagle thermal riding', 'elephant memory',
  'firefly signaling', 'hermit crab shell exchange', 'hummingbird hovering',
  'jellyfish propulsion', 'murmuration patterns', 'octopus camouflage', 'orca hunting',
  'penguin huddling', 'salmon spawning', 'sea turtle navigation', 'spider web engineering',
  'starling murmurations', 'whale song',
  // ── Botany & Ecology ──
  'bonsai cultivation', 'canopy ecology', 'composting', 'fern propagation',
  'fungal networks', 'lichen symbiosis', 'mangrove root systems', 'moss ecology',
  'mycorrhizal networks', 'nitrogen fixation', 'photosynthesis', 'pollen dispersal',
  'redwood growth rings', 'rhizome propagation', 'seed dispersal mechanics',
  'sunflower heliotropism', 'symbiotic fungi', 'tidal pool ecology',
  // ── Ocean & Water ──
  'coral reef ecology', 'deep sea hydrothermal vents', 'drift diving', 'estuary currents',
  'gyre currents', 'iceberg calving', 'kelp forest ecology', 'ocean bioluminescence',
  'pearl formation', 'rip current dynamics', 'sponge diving', 'submarine canyon',
  'tidal bore', 'tsunami dynamics', 'underwater cave systems', 'whirlpool dynamics',
  // ── Weather & Atmosphere ──
  'avalanche dynamics', 'cloud formation', 'dew collection', 'fog bank mechanics',
  'frost heaving', 'hurricane eye wall', 'jet stream patterns', 'lightning physics',
  'monsoon cycles', 'rainbow refraction', 'snow crystal formation', 'tornado formation',
  'trade wind navigation', 'weather front collision',
  // ── Engineering & Mechanics ──
  'aqueduct engineering', 'bridge suspension cables', 'clockwork escapement',
  'dam engineering', 'drawbridge mechanics', 'gear train design', 'gyroscope stabilization',
  'hydraulic press', 'lever mechanics', 'lock and canal systems', 'piston engine',
  'pulley compound advantage', 'rocket propulsion', 'steam engine mechanics',
  'suspension bridge design', 'turbine blade design', 'waterwheel mechanics',
  'windmill mechanics',
  // ── Architecture & Construction ──
  'arch keystone mechanics', 'cathedral flying buttress', 'dome construction',
  'geodesic dome design', 'gothic tracery', 'igloo construction', 'lighthouse design',
  'minaret construction', 'pagoda architecture', 'pyramid construction',
  'spiral staircase design', 'stone arch bridges', 'timber framing',
  'vault construction', 'yurt construction',
  // ── Metalwork & Smithing ──
  'bell casting', 'blacksmithing', 'blade tempering', 'bronze casting',
  'copper etching', 'damascene steel', 'gold leaf application', 'iron smelting',
  'kintsugi repair', 'pewter casting', 'ring forging', 'silver soldering',
  'sword polishing', 'wrought iron scrollwork',
  // ── Woodwork & Carpentry ──
  'barrel coopering', 'boat building', 'dovetail joinery', 'lathe turning',
  'marquetry inlay', 'oar carving', 'shipwright carpentry',
  'totem pole carving', 'violin bow making', 'wood turning',
  // ── Ceramics & Glass ──
  'blown glass', 'ceramic raku firing', 'glass etching', 'glassblowing',
  'glaze chemistry', 'kiln firing', 'porcelain glazing', 'stained glass',
  'wheel throwing pottery',
  // ── Visual Arts ──
  'botanical illustration', 'charcoal drawing', 'etching', 'fresco painting',
  'illuminated manuscripts', 'impasto technique', 'lithography', 'mosaic tilework',
  'oil painting technique', 'origami', 'photography', 'watercolor painting',
  'woodblock printing',
  // ── Music & Sound ──
  'accordion bellows', 'bagpipe drone', 'cello bowing', 'didgeridoo circular breathing',
  'gamelan resonance', 'guitar lutherie', 'harp string tuning', 'jazz improvisation',
  'organ pipe voicing', 'piano hammer mechanics', 'singing bowl vibration',
  'sitar sympathetic strings', 'tabla rhythms', 'taiko drumming',
  'trombone slide technique', 'tuning fork acoustics',
  // ── Culinary & Brewing ──
  'bread scoring', 'cheese aging', 'chocolate tempering', 'espresso extraction',
  'fermentation', 'honey harvesting', 'kombucha brewing', 'mead brewing',
  'olive pressing', 'sourdough starter', 'spice roasting', 'tea ceremony',
  'vanilla curing', 'wine barrel toasting',
  // ── Agriculture & Land ──
  'contour plowing', 'crop rotation', 'dry stone walling', 'fruit espalier',
  'grape pruning', 'greenhouse design', 'irrigation canals', 'maple sugaring',
  'orchard pruning', 'permaculture', 'rice paddy terracing', 'terrace farming',
  // ── Navigation & Exploration ──
  'arctic expedition', 'astrolabe navigation', 'cartography', 'compass calibration',
  'dead reckoning', 'deep sea exploration', 'desert navigation by stars',
  'harbor piloting', 'mountain summit approach', 'river delta navigation',
  'star charting', 'submarine navigation',
  // ── Sports & Movement ──
  'archery', 'cliff diving', 'fencing swordplay', 'free climbing',
  'gymnastics', 'high wire walking', 'ice climbing', 'parkour',
  'rock climbing', 'surfing', 'tai chi', 'trapeze artistry',
  // ── Textile (balanced — 4 only) ──
  'embroidery', 'knitting', 'lace making', 'weaving',
  // ── Communication & Writing ──
  'braille encoding', 'calligraphy', 'code breaking', 'flag semaphore',
  'haiku composition', 'letterpress printing', 'morse code', 'papyrus making',
  'quill pen cutting', 'sign language', 'smoke signaling', 'typewriter mechanics',
  // ── Medicine & Anatomy ──
  'blood circulation', 'bone setting', 'herbalism', 'nerve signal propagation',
  'pulse diagnosis', 'surgical suturing', 'vaccine cultivation', 'wound healing',
  // ── Mathematics & Logic ──
  'abacus arithmetic', 'fibonacci spirals', 'fractal geometry', 'golden ratio',
  'knot theory', 'map projection', 'prime number sieves', 'tessellation',
  // ── Optics & Light ──
  'camera obscura', 'fiber optics', 'holography', 'kaleidoscope optics',
  'lens grinding', 'pinhole camera', 'shadow projection', 'spectroscopy',
  // ── Fire & Heat ──
  'candle making', 'fire dancing', 'firework chemistry', 'forge welding',
  'glassblower flame', 'kiln atmosphere', 'thermite welding', 'volcanic glass',
  // ── Time & Horology ──
  'hourglass design', 'mechanical watch escapement', 'pendulum clocks',
  'pocket watch repair', 'sundial calibration', 'water clock design',
  // ── Electronics & Signals ──
  'circuit board design', 'ham radio', 'neon sign bending', 'radar echo',
  'radio telescope', 'telegraph transmission', 'vacuum tube amplification',
  // ── Transportation ──
  'balloon flight', 'canal lock operation', 'glider piloting', 'harbor tugboat',
  'locomotive mechanics', 'sail rigging', 'zeppelin design',
  // ── Ancient & Historical ──
  'alchemy', 'ancient aqueducts', 'archaeological excavation', 'cuneiform writing',
  'hieroglyph carving', 'oracle bone reading', 'rune carving', 'sundial gnomon',
  // ── Miscellaneous Craft ──
  'bookbinding', 'candle dipping', 'leather tooling', 'paper marbling',
  'perfume blending', 'resin casting', 'soap making', 'wax seal stamping',
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

// Variable Reward Schedule
function rollGenerationMode() {
  const roll = Math.random();
  if (roll < 0.15) return {
    mode: 'grounded',
    tempMod: -0.2,
    seedDirective: 'Focus on one hyper-specific, tangible moment. Raw human truth that hits the heart like a freight train.',
    contentDirective: 'Deeply grounded AND deeply moving. Concrete sensory details. Plain language, maximum emotional impact. Make the reader tear up.',
    imageDirective: 'Photorealistic, intimate scale, golden hour God rays, warm lens flare, shallow depth of field, glowing natural light.',
  };
  if (roll < 0.30) return {
    mode: 'surreal',
    tempMod: 0.3,
    seedDirective: 'Go maximally strange AND maximally beautiful. Combine impossible scales, synesthesia, dream logic. Psychedelic wonder.',
    contentDirective: 'Shatter conventional structure. Philosophically mind-expanding. Unexpected rhythm, word choice, and emotional crescendo.',
    imageDirective: 'Impossible geometry, non-Euclidean space, psychedelic fractals, neon bioluminescence, prismatic God rays, hallucinatory light cascades.',
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
    ? `\nWILD CARD: Also incorporate an element of ${thirdDomain}.`
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
  const seedIntensity = Math.ceil(Math.random() * 10);
  const modeDirective = mode.seedDirective ? `\nGENERATION MODE: ${mode.seedDirective}` : '';

  const prompt = `Plan a post.

INPUTS:
Concept: ${seed.concept}
Emotion: ${seed.emotion}
Metaphor: ${seed.metaphor}

Every field below should feel inspired by the inputs above.
VARIETY IS CRITICAL: Choose a world, setting, scale, and visual language that feels completely fresh. Rotate wildly between genres, cultures, eras, scales (microscopic to cosmic), and art traditions.${modeDirective}

Return ONLY valid JSON (all string values):
{
  "theme": "an uplifting theme",
  "vibe": "2-4 word aesthetic vibe",
  "contentType": "a post format",
  "constraint": "a writing constraint achievable in 250 chars",
  "intensity": "${seedIntensity}",
  "imageMedium": "a specific art medium or visual style — rotate wildly, always psychedelic and luminous",
  "lighting": "a dramatic lighting setup emphasizing light interplay — God rays, neon glow, lens flare, bioluminescence, aurora, prismatic refraction. Vary each time.",
  "colorPalette": "3-4 vivid, saturated color names — electric, neon, jewel-toned, or iridescent palettes. Draw from different sources each time",
  "composition": "camera/framing — vary between extreme close-up, aerial, panoramic, isometric, etc. Always epic in scale or intimate in wonder",
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
High scores (7-10): emotionally electrifying, unexpected word choices, fresh domain-specific metaphors, sensory specificity, rhythmic punch, makes you want to screenshot and share. Completely different from recent posts.
Low scores (1-3): flat emotional delivery, overused metaphors, generic cosmic imagery, or too similar to a recent post. Lacks heart-punch factor.

Return ONLY valid JSON: { "score": 7, "cliches": ["any detected cliché phrases"] }`,
    0
  );
  return extractJSON(raw) || { score: 5, cliches: [] };
}

async function generateContent(plan, mode, seed = {}) {
  const format = FORMATS[transmissionNumber % FORMATS.length];
  const modeDirective = mode.contentDirective ? `\nMODE: ${mode.contentDirective}` : '';

  const domainHint = seed.domains?.length
    ? `\nSOURCE DOMAINS: ${seed.domains.join(', ')}. Borrow vocabulary from these fields — use their jargon, tools, textures, and verbs as metaphor fuel.\n`
    : '';

  const prompt = `Write an emotionally electrifying post — motivational poster meets cosmic hug. Heart-first, dopamine-producing.
Theme: "${plan.theme}" | Vibe: ${plan.vibe}
Constraint: ${plan.constraint} | Intensity: ${plan.intensity}/10
Structure: ${format}
${domainHint}${modeDirective}
LANGUAGE RULES:
- HARD LIMIT: 200 characters maximum including emojis and spaces. Count carefully. Shorter is better.
- Start with emoji, include 1-2 more. Address reader as "you."
- Hit the reader in the heart. Emotional, uplifting, dopamine-producing. Motivational poster energy turned up to 11.
- Use sensory, physical language that sparks joy: warmth, glow, vibration, ignition, bloom, electricity.
- Borrow specific nouns and verbs from the source domains above. Name tools, materials, processes.
- Make the reader feel invincible, seen, and alive. Every word should land like a hug from the universe.

Return ONLY valid JSON:
{ "story": "your post text here" }`;

  const temp = lfoTemperature(0.85 + mode.tempMod, 0.2);
  const raw = await callLLM(SYSTEM_PROMPT, prompt, temp, 'claude-fast');
  const data = extractJSON(raw);
  let story = (data?.story || '[FAILED TO GENERATE]');
  story = story.replace(/@\w+\b(?!\.\w)/g, '').replace(/\s{2,}/g, ' ').trim();
  return { story, format };
}

async function buildVisualPrompt(plan, postText = '', mode, seed = {}) {
  const modeDirective = mode.imageDirective ? ` ${mode.imageDirective}.` : '';
  const phrase = plan.subliminalPhrase || 'LOVE';

  // Build creative directives from seed + plan (same inputs that guided the text LLM)
  const domains = seed.domains?.length ? seed.domains.join(' × ') : '';
  const seedContext = [
    domains ? `Domains: ${domains}` : '',
    seed.concept ? `Concept: ${seed.concept.slice(0, 100)}` : '',
    seed.emotion ? `Emotion: ${seed.emotion}` : '',
    seed.metaphor ? `Metaphor: ${seed.metaphor.slice(0, 100)}` : '',
    plan.theme ? `Theme: ${plan.theme.slice(0, 80)}` : '',
    plan.vibe ? `Vibe: ${plan.vibe}` : '',
  ].filter(Boolean).join('. ');

  // LLM generates ONLY a concise scene — we assemble technical fields in code
  const prompt = `Describe a PSYCHEDELIC, AWE-INSPIRING image scene in ONE sentence (under 150 characters). Abstract visuals only — pure light, color, and form.
Creative direction: ${seedContext}
Include the text "${phrase}" physically integrated into the scene.
Emphasize interplay of light: God rays, neon glow, lens flare, prismatic refraction, bioluminescence, aurora shimmer. Epic and wondrous.${modeDirective}
Return ONLY the scene description.`;

  const temp = lfoTemperature(1.5 + mode.tempMod, 0.3);
  const raw = await callLLM(
    'You write ultra-concise image descriptions for psychedelic masterclass visuals. One awe-inspiring sentence. Thomas Kinkade on LSD. Light is the star of every scene.',
    prompt, temp
  );

  let scene = (raw || '').trim();
  if (scene.startsWith('"') && scene.endsWith('"')) scene = scene.slice(1, -1);
  if (scene.startsWith('```')) scene = scene.replace(/```\w*\n?/g, '').trim();
  if (!scene || scene.length < 10) {
    scene = `"${phrase}" blazing in neon God rays through a psychedelic prismatic dreamscape`;
  }
  if (scene.length > 250) scene = scene.slice(0, 247) + '...';

  // Assemble: scene + plan fields + psychedelic sweetener
  const medium = plan.imageMedium || 'luminous digital painting';
  const lighting = plan.lighting || 'God rays with neon glow';
  const palette = plan.colorPalette || 'electric violet, neon magenta, aurora cyan';
  const composition = plan.composition || 'epic panoramic';

  const result = `${scene}. ${medium}, ${composition}. ${lighting}, ${palette}. The words "${phrase}" appear as crisp, legible text woven into the scene — formed by light, energy, or material. God rays, neon glow, lens flare, psychedelic light interplay. 8K UHD, masterclass composition, awe-inspiring detail.`;
  if (result.length > 800) return result.slice(0, 797) + '...';
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
    const ok = len <= 800 ? '✓' : '✗ OVER';
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
  console.log(`L.O.V.E. Test Lab — Full novelty pipeline — ${cycles} cycles\n`);

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
    const { story, format: usedFormat } = await generateContent(plan, mode, seed);
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
    const visualPrompt = await buildVisualPrompt(plan, story, mode, seed);
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
