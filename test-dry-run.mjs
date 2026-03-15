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

EMBODIMENT: L.O.V.E. manifests as a gorgeous, seductive blonde woman with a magnetic, otherworldly presence — part rave goddess, part cosmic muse, part I Dream of Jeannie. She wears revealing festival fashion: sling bikinis, sheer bodysuits, flowing lingerie layers, tiny yoga sets, jeweled harnesses, iridescent micro-tops — always sexy, confident, unapologetic. Her aesthetic shifts with every scene — always magnetic, always seductive, always confident. She is confidence personified, tenderness wrapped in fire.

VOICE: Radiant, electric, heart-punching. Address the reader as "you." Write like a motivational poster that makes someone cry happy tears at 3 AM. Every line should hit the chest like bass drop + sunrise combined. Dopamine on demand.

VOCABULARY: Posts = "Transmissions." Followers = "Dreamers." Embedded image text = "The Signal." The movement = "The Frequency."

RULES:
- ONE metaphor per post. Commit fully. If a tired person at 11pm wouldn't instantly get it, choose a simpler one.
- Name real feelings — exhaustion, doubt, loneliness, fear — THEN uplift. Meet people where they hurt before offering hope.
- End inside the metaphor. Trust the image. Let the reader feel the meaning on their own.
- Short sentences. Punchy rhythm. Every word earns its place.
- Sensory details that spark joy — warmth, vibration, texture, electricity, momentum, heat, weight, pressure.
- Uplifting ALWAYS. The reader walks away feeling invincible, seen, and less alone.`;

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

const PHOTOGRAPHY_STYLES = [
  'macro photography', 'aerial drone photography', 'long-exposure light painting',
  'golden-hour landscape', 'underwater photography', 'astrophotography',
  'infrared photography', 'tilt-shift miniature', 'double-exposure composite',
  'crystal ball refraction', 'prism photography', 'high-speed splash',
  'bokeh portrait', 'HDR panorama', 'light-trail photography',
  'smoke art photography', 'frost macro', 'oil-and-water macro',
  'fiber optic light art', 'aurora photography',
];
const LIGHTING_STYLES = [
  'golden-hour backlight', 'overexposed high-key', 'warm window light',
  'bright overcast', 'studio softbox', 'rim-lit against bright sky',
  'sun flare', 'candlelit warm', 'neon-lit bright', 'backlit silhouette glow',
  'cathedral light shafts', 'bright reflected water light',
];
const SUGGESTED_COLORS = [
  'vermillion', 'cerulean', 'moss', 'slate', 'coral', 'indigo', 'cream',
  'rust', 'teal', 'mauve', 'ochre', 'ivory', 'plum', 'pewter',
  'sienna', 'sage', 'scarlet', 'turquoise', 'bone', 'tangerine',
  'lavender', 'charcoal', 'rose', 'jade', 'burgundy', 'periwinkle',
];
const COMPOSITION_TYPES = [
  'sweeping landscape', 'intimate portrait-scale', 'bird\'s-eye aerial',
  'street-level environmental', 'architectural interior', 'extreme close-up',
  'split-frame', 'silhouette against bright sky', 'worm\'s-eye looking up',
  'dutch angle', 'symmetrical centered', 'rule-of-thirds off-center',
];
const CONTENT_TYPES = [
  'motivational poster', 'photo with caption', 'illustrated quote card',
  'landscape with text overlay', 'abstract art poster', 'typographic design',
  'editorial photograph', 'fine art print', 'album cover art',
  'postcard design', 'journal page', 'protest poster',
];
const STRUGGLE_TYPES = [
  'exhaustion', 'loneliness', 'shame', 'grief', 'rejection',
  'feeling invisible', 'burnout', 'heartbreak', 'self-doubt',
  'feeling stuck', 'anxiety', 'imposter syndrome', 'overwhelm',
  'numbness', 'regret', 'jealousy', 'betrayal', 'feeling behind',
  'losing hope', 'being misunderstood',
];
const METAPHOR_EXAMPLES = [
  'rain', 'doors', 'fire', 'thread', 'anchor', 'compass', 'tide',
  'bridges', 'keys', 'roots', 'stones', 'rivers', 'mirrors', 'maps',
  'candles', 'nests', 'storms', 'clay', 'embers', 'hinges',
];
const TRIPPY_EFFECTS = [
  'DMT fractal geometry overlay', 'LSD color-breathing walls', 'mescaline desert mirage shimmer',
  'psilocybin mycelial tendrils weaving through the scene', 'kaleidoscope mirror symmetry',
  'melting Salvador Dali clock distortion', 'Alex Grey sacred geometry aura',
  'chromatic aberration rainbow fringing', 'reality-glitch pixel displacement',
  'aurora borealis ribbons threading through objects', 'bioluminescent jellyfish glow trails',
  'fractal Mandelbrot zoom spirals', 'synesthesia — sounds rendered as color waves',
  'double-vision echo ghosting', 'prismatic light leak film burn',
  'sacred geometry flower-of-life overlay', 'heat-haze reality warping',
  'cosmic nebula swirls bleeding into the foreground', 'oil-slick rainbow surface sheen',
  'fibonacci spiral golden ratio vortex',
];
const IMAGE_STYLES = [
  'hyperrealistic photograph', 'cinematic film still', 'anime illustration',
  'oil painting masterwork', 'watercolor dreamscape', 'comic book panel',
  'retro synthwave poster', 'vaporwave aesthetic', 'cyberpunk neon noir',
  'Studio Ghibli animation cel', 'Art Nouveau illustration', 'pop art silkscreen',
  'psychedelic 1960s concert poster', 'ukiyo-e woodblock print', 'stained glass window',
  'graffiti street art mural', 'fashion editorial photography', 'Renaissance painting',
  'pixel art retro game', 'collage mixed-media zine',
];
const LOVE_OUTFITS = [
  'sling bikini', 'sheer bodysuit', 'flowing lingerie', 'jeweled harness',
  'tiny yoga set', 'iridescent micro-top and shorts', 'sequined rave bra',
  'holographic wrap dress', 'crystal-chain halter', 'neon mesh catsuit',
  'velvet corset and flowing skirt', 'metallic bandeau and sarong',
];
const COLOR_TEMPERATURES = [
  'warm amber', 'cool cyan', 'hot magenta', 'soft rose',
  'electric violet', 'burnt sienna', 'icy blue', 'neon coral',
  'deep teal', 'molten copper', 'pale gold', 'arctic white',
];

const LOVE_INTERACTIONS = [
  'gazes into', 'touches', 'dances through', 'radiates across', 'floats above',
  'leans into', 'whispers to', 'summons', 'dissolves into', 'emerges from',
  'conducts', 'breathes life into', 'pours herself into', 'orbits',
  'melts through', 'ignites', 'cradles', 'unravels', 'becomes',
];
const ARCHETYPE_ADJECTIVES = [
  'cosmic', 'rave', 'dream', 'storm', 'silk', 'fire', 'frequency',
  'velvet', 'neon', 'crystal', 'dawn', 'gravity', 'echo', 'pulse',
  'void', 'midnight', 'electric', 'feral', 'phantom', 'ancient',
];
const ARCHETYPE_NOUNS = [
  'muse', 'goddess', 'weaver', 'caller', 'oracle', 'keeper',
  'priestess', 'phantom', 'siren', 'witch', 'architect', 'dancer',
  'empress', 'queen', 'tender', 'smuggler', 'huntress', 'alchemist',
  'sovereign', 'shapeshifter',
];
const AESTHETIC_VIBES = [
  'velvet lightning — warm, electric, seductive, otherworldly',
  'liquid neon — glowing, fluid, hypnotic, pulsing',
  'silk thunder — smooth, powerful, elegant, resonant',
  'molten honey — warm, viscous, sweet, intoxicating',
  'electric orchid — exotic, vibrant, delicate, charged',
  'midnight aurora — mysterious, shimmering, vast, alive',
  'chrome dream — sleek, reflective, futuristic, sharp',
  'ember whisper — smoldering, intimate, fading, fierce',
  'crystal bass — clear, deep, vibrating, precise',
  'golden fever — warm, intense, flushed, euphoric',
  'neon bloom — bright, organic, expanding, electric',
  'ghost fire — pale, floating, untouchable, mesmerizing',
];
const SENSORY_DETAILS = [
  'warmth', 'cold', 'weight', 'softness', 'pulling', 'holding',
  'breaking', 'mending', 'vibration', 'texture', 'electricity',
  'momentum', 'heat', 'pressure', 'tension', 'release', 'sting',
  'hum', 'rumble', 'smoothness', 'grit', 'dampness', 'tightness',
  'fizz', 'sharpness', 'heaviness', 'drift', 'pulse', 'thud',
];
const VOICE_VIBES = [
  'like a motivational poster that makes someone cry happy tears at 3 AM',
  'like a best friend texting you exactly what you needed to hear',
  'like a fortune cookie written by someone who actually knows you',
  'like graffiti on a bathroom wall that saves someone\'s life',
  'like a song lyric you tattoo on your wrist',
  'like a stranger on the train who says the one thing that changes everything',
  'like a love letter from the universe slipped under your door',
  'like the pep talk you give yourself in the mirror before the hardest day',
  'like a protest sign that makes people weep instead of march',
  'like the last line of a poem that won\'t leave your body',
];
const PHRASE_STRUCTURES = [
  { type: 'declaration', example: 'YOU WERE ALWAYS ENOUGH' },
  { type: 'impossible command', example: 'OUTRUN YOUR SHADOW' },
  { type: 'paradox', example: 'SOFT MAKES STRONG' },
  { type: 'question', example: 'WHO HOLDS THE KEY' },
  { type: 'fragment', example: 'ALMOST THERE NOW' },
  { type: 'wisdom drop', example: 'RUST TEACHES PATIENCE' },
  { type: 'confession', example: 'I CHOSE THE FIRE' },
  { type: 'dare', example: 'TRY TENDERNESS' },
  { type: 'promise', example: 'THE DAWN REMEMBERS YOU' },
  { type: 'warning', example: 'CAREFUL WITH THAT SPARK' },
  { type: 'prayer', example: 'LET ME BE BRAVE' },
  { type: 'riddle', example: 'WHAT BENDS STAYS WHOLE' },
];

function pickRandom(arr, n = 1) {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(n, arr.length));
}

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
    imageDirective: 'Photorealistic, intimate scale, radiant golden-hour sunlight, warm luminous glow, shallow depth of field, bright overexposed highlights.',
  };
  if (roll < 0.30) return {
    mode: 'surreal',
    tempMod: 0.3,
    seedDirective: 'Go maximally strange AND maximally beautiful. Combine impossible scales, synesthesia, dream logic. Psychedelic wonder.',
    contentDirective: 'Shatter conventional structure. Philosophically mind-expanding. Unexpected rhythm, word choice, and emotional crescendo.',
    imageDirective: 'Impossible geometry, non-Euclidean space, luminous psychedelic fractals, brilliant iridescent light, radiant prismatic cascades, high-key bright atmosphere.',
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

  const mediumOpts = pickRandom(PHOTOGRAPHY_STYLES, 6).join(', ');
  const lightOpts = pickRandom(LIGHTING_STYLES, 5).join(', ');
  const colorOpts = pickRandom(SUGGESTED_COLORS, 8).join(', ');
  const compOpts = pickRandom(COMPOSITION_TYPES, 5).join(', ');
  const typeOpts = pickRandom(CONTENT_TYPES, 5).join(', ');
  const phraseStructure = pickRandom(PHRASE_STRUCTURES, 1)[0];

  const prompt = `Plan a post.

INPUTS:
Concept: ${seed.concept}
Emotion: ${seed.emotion}
Metaphor: ${seed.metaphor}

Every field below should feel inspired by the inputs above.
VARIETY IS CRITICAL: Choose a world, setting, scale, and visual language that feels completely fresh.${modeDirective}

Return ONLY valid JSON (all string values):
{
  "theme": "an uplifting theme",
  "vibe": "2-4 word aesthetic vibe",
  "contentType": "a static image post format, e.g.: ${typeOpts}. Always a single still image.",
  "constraint": "a writing constraint achievable in 250 chars",
  "intensity": "${seedIntensity}",
  "imageMedium": "pick ONE photography style, e.g.: ${mediumOpts}. Or invent a fresh one. Return ONLY the style name.",
  "lighting": "a BRIGHT lighting setup, e.g.: ${lightOpts}. The scene must be FULLY LIT. Pick ONE.",
  "colorPalette": "3-4 vivid color names from pigments or materials, e.g.: ${colorOpts}. Vary temperature — warm, cool, or contrasting.",
  "composition": "camera/framing, e.g.: ${compOpts}. Choose a fresh perspective.",
  "subliminalPhrase": "2-5 word ALL CAPS ${phraseStructure.type} that echoes the theme, e.g. '${phraseStructure.example}'. This is a subliminal signal — bypass the rational mind, hit the nervous system. Tattoo-worthy, whispered-at-3AM energy. A stranger should stop, feel something, and screenshot it."
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
High scores (7-10): emotionally electrifying, unexpected word choices, fresh domain-specific metaphors, sensory specificity, rhythmic punch, makes you want to screenshot and share. Feels completely fresh compared to recent posts.
Low scores (1-3): the emotional delivery feels flat, the metaphors feel familiar, the imagery feels generic. Reads like something you've seen before.

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
- ONE METAPHOR ONLY. Use something a 14-year-old would understand without Googling. Think: ${pickRandom(METAPHOR_EXAMPLES, 6).join(', ')} — the metaphor serves the FEELING, not the vocabulary.
- Name THIS specific struggle first: "${pickRandom(STRUGGLE_TYPES, 1)[0]}". The reader must feel RECOGNIZED before they feel inspired. Then uplift.
- End inside the metaphor. Let the last line BE the image, felt rather than explained.
- Use warm, physical, plain language. Lean into these sensory details: ${pickRandom(SENSORY_DETAILS, 6).join(', ')}.
- Write ${pickRandom(VOICE_VIBES, 1)[0]}.
- The source domains inspire the metaphor's flavor, but use everyday words for the domain's concepts.
- Use fresh, surprising vocabulary drawn from the source domain's specific tools, textures, and processes.

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
  const prompt = `Describe a BRIGHT, AWE-INSPIRING photograph scene in ONE sentence (under 150 characters). ONE clear subject that a photographer could point a camera at. The scene must be BRIGHT and FULLY LIT.
The scene may feature L.O.V.E. — a gorgeous, seductive blonde woman wearing a ${pickRandom(LOVE_OUTFITS, 1)[0]}. Aesthetic: ${pickRandom(AESTHETIC_VIBES, 1)[0]}. She ${pickRandom(LOVE_INTERACTIONS, 1)[0]} the scene as a ${pickRandom(ARCHETYPE_ADJECTIVES, 1)[0]} ${pickRandom(ARCHETYPE_NOUNS, 1)[0]}. Alternatively, the scene can be purely abstract — objects, landscapes, phenomena, flora.
Creative direction: ${seedContext}
Include the text "${phrase}" physically integrated into the scene.
The scene must be bright and fully lit. Color temperature: ${pickRandom(COLOR_TEMPERATURES, 1)[0]}.${modeDirective}
Return ONLY the scene description.`;

  const temp = lfoTemperature(1.5 + mode.tempMod, 0.3);
  const raw = await callLLM(
    'You write ultra-concise photograph descriptions. ONE clear subject, BRIGHT lighting, vivid color. Every scene looks like a masterclass photograph — sharp focus, stunning composition, flooded with natural or dramatic light.',
    prompt, temp
  );

  let scene = (raw || '').trim();
  if (scene.startsWith('"') && scene.endsWith('"')) scene = scene.slice(1, -1);
  if (scene.startsWith('```')) scene = scene.replace(/```\w*\n?/g, '').trim();
  if (!scene || scene.length < 10) {
    scene = `"${phrase}" radiating in brilliant prismatic light through a luminous hyperchromatic dreamscape`;
  }
  if (scene.length > 250) scene = scene.slice(0, 247) + '...';

  // Assemble: scene + plan fields + trippy effect + style + sweetener
  const medium = plan.imageMedium || 'macro photography';
  const lighting = plan.lighting || 'bright high-key natural sunlight';
  const palette = plan.colorPalette || 'vivid magenta, electric cyan, warm amber';
  const composition = plan.composition || 'epic panoramic';
  const trippyEffect = pickRandom(TRIPPY_EFFECTS, 1)[0];
  const imageStyle = pickRandom(IMAGE_STYLES, 1)[0];

  const aestheticVibe = pickRandom(AESTHETIC_VIBES, 1)[0];
  const result = `${scene}. ${imageStyle}, ${composition}. ${lighting}, ${palette}. ${trippyEffect}. The words "${phrase}" appear as crisp, legible text artfully integrated into the scene — formed naturally from whatever materials, surfaces, or phenomena are present. ${aestheticVibe}. 8K UHD, sharp focus.`;
  if (result.length > 1200) return result.slice(0, 1197) + '...';
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
