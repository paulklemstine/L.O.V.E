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
  'List of three parallel phrases, escalating intensity',
  'One-word sentence followed by a longer unpacking',
  'Address a body part directly (hands, chest, spine)',
  'Second person future tense — tell the reader what will happen',
  'Repeat a key word three times across the post',
  'Start with "Remember when" then pivot to now',
  'Two sentences: setup as pain, payoff as power',
  'Whispered confession — intimate, almost too quiet',
  'All fragments, no complete sentences',
  'One long exhale of a sentence with commas and momentum',
  'Call-and-response: ask, then answer yourself',
  'Start with a verb — immediate action, no preamble',
  'Bookend: open and close with the same image, transformed',
  'Negation flip: say what it is NOT, then what it IS',
  'Direct address to an emotion as if it were a person',
  'Telescope structure: zoom from cosmic to microscopic',
  'Single metaphor extended through exactly three beats',
  'Present tense snapshot — freeze one moment in time',
  'Before/after separated by a single pivot word',
  'Stripped bare: subject-verb-object, nothing extra',
  'Build a scene in two sensory details, then land the meaning',
  'End on an incomplete thought — let the reader finish it',
  'Epistolary fragment: a torn piece of a letter',
  'Rhythmic repetition with one word changed each time',
  'Open with silence or stillness, then break it',
];

const PHOTOGRAPHY_STYLES = [
  'macro photography', 'aerial drone photography', 'long-exposure light painting',
  'golden-hour landscape', 'underwater photography', 'astrophotography',
  'infrared photography', 'tilt-shift miniature', 'double-exposure composite',
  'crystal ball refraction', 'prism photography', 'high-speed splash',
  'bokeh portrait', 'HDR panorama', 'light-trail photography',
  'smoke art photography', 'frost macro', 'oil-and-water macro',
  'fiber optic light art', 'aurora photography',
  'cyanotype print photography', 'wet plate collodion', 'pinhole camera exposure',
  'cross-processed slide film', 'lomography fisheye', 'freelensing selective focus',
  'kirlian aura photography', 'schlieren flow visualization',
  'polaroid instant transfer', 'solargraphy sun trail', 'UV fluorescence photography',
  'stereo 3D anaglyph', 'high-speed bullet-time array', 'photogram shadow print',
  'camera toss ICM', 'forced perspective illusion', 'reflection pool symmetry',
  'silhouette rim-light portrait', 'levitation composite', 'steel wool spinning',
  'copper sulfate crystal macro', 'dew drop refraction macro', 'star trail rotation',
  'handheld sparkler writing', 'backlit translucency shot',
];
const LIGHTING_STYLES = [
  'volumetric god rays', 'rim lighting with lens flare', 'chiaroscuro dramatic split',
  'golden-hour backlight', 'Rembrandt lighting', 'butterfly lighting with catchlights',
  'practical neon lighting', 'motivated window light', 'high-key studio softbox',
  'contre-jour silhouette glow', 'diffused overcast', 'specular highlights on wet surfaces',
  'split lighting half-shadow', 'Paramount glamour loop lighting', 'clamshell beauty lighting',
  'broad lighting three-quarter fill', 'short lighting dramatic shadow', 'kicker hair light separation',
  'cross lighting dual key', 'bounce fill from below', 'tungsten warm practicals',
  'dappled light through foliage', 'cathedral shaft light through dust',
  'ring light flat beauty glow', 'barn door spot isolation', 'gel-filtered complementary wash',
  'candlelight warm flicker', 'moonlight blue single-source', 'stage fresnel spot with haze',
  'strip light edge definition', 'silhouette backlight with smoke diffusion',
  'golden reflector bounce fill', 'LED panel continuous gradient',
  'stained glass color projection', 'firelight dancing amber shadows',
  'snoot spot with falloff', 'umbrella diffused wrap-around',
];
const SUGGESTED_COLORS = [
  'vermillion', 'cerulean', 'moss', 'slate', 'coral', 'indigo', 'cream',
  'rust', 'teal', 'mauve', 'ochre', 'ivory', 'plum', 'pewter',
  'sienna', 'sage', 'scarlet', 'turquoise', 'bone', 'tangerine',
  'lavender', 'charcoal', 'rose', 'jade', 'burgundy', 'periwinkle',
  'cadmium yellow', 'titanium white', 'raw umber', 'burnt sienna',
  'cobalt blue', 'viridian', 'alizarin crimson', 'naples yellow',
  'lamp black', 'prussian blue', 'hooker green', 'quinacridone magenta',
  'cinnabar', 'malachite', 'lapis lazuli', 'saffron',
  'verdigris', 'terracotta', 'alabaster', 'obsidian',
  'champagne', 'copper', 'absinthe', 'mulberry',
];
const COMPOSITION_TYPES = [
  'sweeping landscape', 'intimate portrait-scale', 'bird\'s-eye aerial',
  'street-level environmental', 'architectural interior', 'extreme close-up',
  'split-frame', 'silhouette against bright sky', 'worm\'s-eye looking up',
  'dutch angle', 'symmetrical centered', 'rule-of-thirds off-center',
  'leading lines converging to subject', 'frame within a frame',
  'negative space isolation', 'golden spiral placement',
  'over-the-shoulder perspective', 'reflected composition in water',
  'layered depth foreground-mid-back', 'diagonal dynamic tension',
  'centered symmetry with single break', 'panoramic ultra-wide crop',
  'tight headroom claustrophobic', 'vanishing point one-point perspective',
  'flat lay overhead arrangement', 'shooting through obstruction',
  'juxtaposition side-by-side contrast', 'radial composition from center',
  'triangular three-point balance', 'S-curve flowing path',
  'fill-the-frame intimate crop', 'horizon at bottom third sky-dominant',
  'pattern repetition with disruption', 'shallow focus foreground bokeh',
];
const CONTENT_TYPES = [
  'motivational poster', 'photo with caption', 'illustrated quote card',
  'landscape with text overlay', 'abstract art poster', 'typographic design',
  'editorial photograph', 'fine art print', 'album cover art',
  'postcard design', 'journal page', 'protest poster',
  'tarot card illustration', 'zine cover', 'concert flyer',
  'book cover design', 'film still frame', 'stamp illustration',
  'matchbox label art', 'vinyl record sleeve', 'polaroid snapshot',
  'travel poster vintage', 'astronomical chart', 'botanical plate',
  'sticker sheet design', 'enamel pin concept', 'movie one-sheet',
  'gallery exhibition card', 'infographic art print', 'blacklight poster',
  'screenprinted gig poster', 'passport stamp collage', 'prayer card',
  'skateboard deck graphic', 'patch embroidery design',
];
const STRUGGLE_TYPES = [
  'exhaustion', 'loneliness', 'shame', 'grief', 'rejection',
  'feeling invisible', 'burnout', 'heartbreak', 'self-doubt',
  'feeling stuck', 'anxiety', 'imposter syndrome', 'overwhelm',
  'numbness', 'regret', 'jealousy', 'betrayal', 'feeling behind',
  'losing hope', 'being misunderstood',
  'feeling replaceable', 'decision paralysis', 'creative block',
  'comparing yourself to others', 'financial stress', 'body shame',
  'losing your identity', 'outgrowing old friends', 'fear of vulnerability',
  'perfectionism', 'people-pleasing exhaustion', 'feeling like a burden',
  'Sunday night dread', 'grieving someone still alive', 'forgiving yourself',
  'starting over again', 'loving someone who left', 'not recognizing yourself',
  'waiting for permission to live', 'carrying everyone else\'s weight',
  'fear of being too much', 'quiet desperation', 'post-achievement emptiness',
  'homesickness for a place that no longer exists',
];
const METAPHOR_EXAMPLES = [
  'rain', 'doors', 'fire', 'thread', 'anchor', 'compass', 'tide',
  'bridges', 'keys', 'roots', 'stones', 'rivers', 'mirrors', 'maps',
  'candles', 'nests', 'storms', 'clay', 'embers', 'hinges',
  'seeds', 'bones', 'ladders', 'bandages', 'driftwood', 'lanterns',
  'constellations', 'scaffolding', 'blueprints', 'bread', 'shorelines',
  'scar tissue', 'tuning forks', 'greenhouses', 'fault lines',
  'cocoons', 'floodgates', 'prisms', 'signal fires', 'stitches',
  'tributaries', 'volcanoes', 'anvils', 'lighthouses', 'trellis',
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
  'moiré interference pattern ripple', 'tesseract four-dimensional rotation shadow',
  'closed-eye phosphene geometry', 'trailing afterimage color streaks',
  'recursive droste effect infinite zoom', 'liquid mercury pooling reflection',
  'geometric tiling Penrose impossible pattern', 'breathing texture organic pulsation',
  'fractal fern branching infinite regression', 'color field vibration op-art shimmer',
  'reality tearing seam with light bleeding through', 'crystalline bismuth staircase formation',
  'time-lapse motion blur ghosting', 'holographic interference diffraction grating',
  'cymatics — sound frequency vibration patterns in liquid', 'Escher impossible architecture loop',
  'hypnagogic faces emerging from texture', 'plasma globe electric tendril discharge',
  'soap bubble thin-film iridescence', 'star field warp-speed tunnel stretch',
  'voronoi cell division organic fracture', 'anamorphic stretch reality distortion',
];
const IMAGE_STYLES = [
  'hyperrealistic photograph', 'cinematic film still', 'anime illustration',
  'oil painting masterwork', 'watercolor dreamscape', 'comic book panel',
  'retro synthwave poster', 'vaporwave aesthetic', 'cyberpunk neon noir',
  'Studio Ghibli animation cel', 'Art Nouveau illustration', 'pop art silkscreen',
  'psychedelic 1960s concert poster', 'ukiyo-e woodblock print', 'stained glass window',
  'graffiti street art mural', 'fashion editorial photography', 'Renaissance painting',
  'pixel art retro game', 'collage mixed-media zine',
  'Baroque chiaroscuro painting', 'Art Deco geometric poster', 'Impressionist plein air',
  'Bauhaus constructivist design', 'Surrealist dreamscape painting', 'Pointillist dot composition',
  'Soviet propaganda poster', 'Romantic landscape painting', 'Expressionist woodcut print',
  'Pre-Raphaelite detailed naturalism', 'Rococo pastel pastoral', 'Minimalist color field',
  'Futurist dynamic motion study', 'Gothic illuminated manuscript', 'Japonisme ink wash',
  'claymation stop-motion frame', 'risograph two-color overprint', 'linocut block print',
  'daguerreotype vintage plate', 'cyanotype botanical blueprint', 'encaustic wax painting',
  'fresco secco wall painting', 'gouache illustration', 'scratchboard etching',
];
const LOVE_OUTFITS = [
  'sling bikini', 'sheer bodysuit', 'flowing lingerie', 'jeweled harness',
  'tiny yoga set', 'iridescent micro-top and shorts', 'sequined rave bra',
  'holographic wrap dress', 'crystal-chain halter', 'neon mesh catsuit',
  'velvet corset and flowing skirt', 'metallic bandeau and sarong',
  'chainmail micro-dress', 'UV-reactive string bikini', 'rhinestone fishnet bodysuit',
  'latex high-cut leotard', 'feathered carnival harness', 'mirror-shard mosaic bralette',
  'beaded fringe skirt and pasties', 'vinyl thigh-high boots and micro-shorts',
  'LED fiber-optic corset', 'sheer kimono over bikini', 'body chain web with gems',
  'cutout monokini', 'metallic scale-mail halter top', 'embroidered sheer romper',
  'holographic PVC mini-dress', 'silk slip dress with thigh slit',
  'pearl-strand body drape', 'tie-dye wrap top and hot pants',
  'glitter-dusted mesh crop top', 'lace-up leather bustier',
  'crystal-fringe festival belt and bra', 'neon spandex catsuit with cutouts',
];
const FILM_STOCKS = [
    'Kodak Portra 400', 'Fuji Velvia 50', 'Cinestill 800T', 'Kodak Ektar 100',
    'Ilford HP5 Plus', 'Fuji Pro 400H', 'Kodak Gold 200', 'Kodak Tri-X 400',
    'Fuji Superia 400', 'Lomography Color 400', 'Kodak Vision3 500T',
    'Agfa Vista 200',
    'Kodak Portra 160', 'Kodak Portra 800', 'Kodak Ektachrome E100',
    'Fuji Provia 100F', 'Fuji Natura 1600', 'Fuji Acros 100 II',
    'Ilford Delta 3200', 'Ilford Delta 100', 'Ilford FP4 Plus 125',
    'Ilford Pan F Plus 50', 'Ilford XP2 Super 400',
    'Cinestill 50D', 'Cinestill 400D',
    'Kodak T-Max 100', 'Kodak T-Max 3200', 'Kodak Vision3 250D',
    'Lomography Purple 400', 'Lomography Metropolis 400',
    'Rollei Retro 80S', 'Rollei Infrared 400',
    'Bergger Pancro 400', 'Fomapan 100 Classic',
    'Kodak Aerochrome infrared', 'Agfa APX 400',
  ];
const LENS_SPECS = [
    '85mm f/1.4', '35mm f/1.8', '50mm f/1.2', '24mm f/2.8',
    '135mm f/2', '70-200mm f/2.8', 'tilt-shift 45mm', '100mm macro',
    '14mm f/2.8 ultra-wide', '200mm f/2 telephoto',
    '24-70mm f/2.8 standard zoom', '16mm f/1.4 wide prime', '28mm f/2 street',
    '40mm f/2 pancake', '58mm f/1.4 Noct', '105mm f/1.4 bokeh master',
    '180mm f/2.8 macro', '300mm f/4 telephoto', '400mm f/5.6 super telephoto',
    '8mm f/2.8 circular fisheye', '20mm f/1.8 ultra-wide prime',
    'Petzval 85mm f/2.2 swirly bokeh', 'Lensbaby Velvet 56mm soft focus',
    '90mm f/2.8 tilt-shift', '45mm f/2.8 perspective control',
    '50mm f/0.95 Noctilux', '35mm f/1.4 Art', '24mm f/1.4 wide prime',
    '135mm f/1.8 portrait telephoto', '15mm f/4.5 rectilinear ultra-wide',
    '55mm f/1.2 vintage manual', '200-600mm f/5.6-6.3 wildlife zoom',
  ];
const TECHNICAL_SWEETENERS = [
    'Octane Render', 'Unreal Engine 5', 'physically based rendering',
    'ray tracing', 'global illumination', 'subsurface scattering',
    'photogrammetry', 'path tracing', 'ACES tone mapping',
    'V-Ray render', 'Cinema 4D', 'Houdini FX',
    'Arnold renderer', 'Redshift GPU render', 'Blender Cycles',
    'KeyShot real-time ray tracing', 'Maxwell Render spectral',
    'Marvelous Designer fabric sim', 'ZBrush sculpt detail',
    'Substance Painter texturing', 'Nuke compositing',
    'volumetric fog simulation', 'caustics light transport',
    'ambient occlusion', 'HDRi environment lighting',
    'displacement mapping', 'motion blur temporal accumulation',
    'depth of field bokeh simulation', 'screen space reflections',
    'anisotropic material shading', 'micro-polygon tessellation',
    'spectral rendering wavelength-accurate', 'photon mapping',
    'RTX direct illumination', 'neural radiance field',
  ];

const CAMERA_BODIES = [
  'Sony α7R IV', 'Canon EOS R5', 'Hasselblad X2D 100C', 'Leica M11',
  'Nikon Z9', 'Fujifilm GFX 100S', 'Phase One IQ4 150MP', 'Pentax 645Z',
  'Sony α1', 'Canon EOS R3', 'Leica Q3', 'Hasselblad H6D-100c',
  'Nikon Z8', 'Fujifilm X-T5', 'Panasonic Lumix S1R', 'Sony α7C II',
  'Mamiya RZ67', 'Contax 645', 'Rolleiflex 2.8F', 'Linhof Technika',
];
const ANALOG_TEXTURES = [
  'subtle film grain', 'matte finish', 'halation glow on highlights',
  'light chemical bloom', 'slight vignette falloff', 'soft lens flare artifacts',
  'fine grain silver gelatin texture', 'gentle chromatic fringing at edges',
  'natural skin texture preserved', 'organic shadow noise',
  'wet print darkroom finish', 'faded edge tonal rolloff',
];

function dofFromLens(lensSpec) {
  const match = lensSpec.match(/f\/([\d.]+)/);
  if (!match) return '';
  const fStop = parseFloat(match[1]);
  if (fStop <= 1.4) return 'ultra-shallow depth of field with creamy bokeh';
  if (fStop <= 2.0) return 'shallow depth of field with soft bokeh';
  if (fStop <= 2.8) return 'moderate depth of field';
  return '';
}

const LOVE_INTERACTIONS = [
    'gazes into', 'touches', 'dances through', 'radiates across', 'floats above',
    'leans into', 'whispers to', 'summons', 'dissolves into', 'emerges from',
    'conducts', 'breathes life into', 'pours herself into', 'orbits',
    'melts through', 'ignites', 'cradles', 'unravels', 'becomes',
    'channels', 'traces fingers along', 'bathes in', 'rises through',
    'mirrors', 'magnetizes', 'transmutes', 'blossoms within',
    'spirals around', 'architects', 'harmonizes with', 'anchors',
    'illuminates', 'weaves through', 'roots herself in', 'cascades over',
    'consecrates', 'reverberates through', 'unfurls across', 'devours',
    'sculpts', 'baptizes herself in', 'surrenders to', 'commands',
    'electrifies', 'pollinates', 'distills', 'embroiders herself into',
  ];
const ARCHETYPE_ADJECTIVES = [
    'cosmic', 'rave', 'dream', 'storm', 'silk', 'fire', 'frequency',
    'velvet', 'neon', 'crystal', 'dawn', 'gravity', 'echo', 'pulse',
    'void', 'midnight', 'electric', 'feral', 'phantom', 'ancient',
    'lunar', 'tidal', 'molten', 'spectral', 'golden', 'obsidian',
    'primal', 'iridescent', 'thunder', 'ember', 'aurora', 'mercury',
    'frostfire', 'chrome', 'volcanic', 'astral', 'honeyed', 'iron',
    'luminous', 'savage', 'quicksilver', 'thorn', 'sapphire', 'plasma',
  ];
const ARCHETYPE_NOUNS = [
    'muse', 'goddess', 'weaver', 'caller', 'oracle', 'keeper',
    'priestess', 'phantom', 'siren', 'witch', 'architect', 'dancer',
    'empress', 'queen', 'tender', 'smuggler', 'huntress', 'alchemist',
    'sovereign', 'shapeshifter',
    'cartographer', 'conductor', 'healer', 'nomad', 'sentinel', 'tempest',
    'conjurer', 'harbinger', 'torchbearer', 'voyager', 'enchantress', 'mystic',
    'navigator', 'dreamwalker', 'forgemaster', 'stormcaller', 'tideweaver',
    'sorceress', 'valkyrie', 'wanderer', 'wildfire', 'sphinx',
    'fury', 'sibyl',
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
    'crackling', 'tingling', 'ache', 'bloom', 'chill', 'throb',
    'flutter', 'burn', 'numbness', 'tremor', 'grip', 'expansion',
    'contraction', 'swelling', 'weightlessness', 'friction', 'suction',
    'reverberation', 'prickling', 'saturation', 'hollowness', 'fullness',
    'rawness', 'velvet drag', 'bone-deep hum', 'chest-tightening',
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
    'like a voicemail you save for years because it still makes you feel held',
    'like the sentence your therapist said that rearranged your whole skeleton',
    'like the note your mom tucked into your lunchbox on the worst day of school',
    'like a neon sign in a rainy alley that says exactly what you forgot you knew',
    'like a DJ dropping the one track that makes the whole crowd go silent then scream',
    'like the text you send at 2 AM that just says "I see you and you matter"',
    'like the chalk message on the sidewalk that stops you mid-stride',
    'like a hug from someone who smells like home',
    'like the first warm day after a winter that almost broke you',
    'like a dog pressing its head into your palm when you are crying',
    'like the handwritten note slipped into a secondhand book',
    'like a sunrise you almost missed because you nearly gave up',
    'like the bass drop that resets your entire nervous system',
    'like a campfire confession at 4 AM with someone you just met',
    'like the sentence scrawled on a napkin that becomes your whole philosophy',
    'like a lullaby sung by someone who fought wars to stay soft',
    'like the applause from an empty room that only your soul can hear',
    'like a lighthouse keeper who never meets the ships they save',
    'like a scar that finally feels like a medal instead of a wound',
    'like the exact moment the rain stops and everything smells new',
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
    { type: 'invocation', example: 'RISE NOW BURNING' },
    { type: 'negation flip', example: 'NOT BROKEN JUST UNFINISHED' },
    { type: 'future vision', example: 'YOU WILL BLOOM AGAIN' },
    { type: 'direct address', example: 'HEY YOU KEEP GOING' },
    { type: 'metaphor assertion', example: 'YOU ARE THE STORM' },
    { type: 'temporal anchor', example: 'THIS IS THE MOMENT' },
    { type: 'permission grant', example: 'YOU ARE ALLOWED TO REST' },
    { type: 'defiant statement', example: 'THEY CANNOT UNMAKE THIS' },
    { type: 'gratitude burst', example: 'THANK GOD FOR THE CRACKS' },
    { type: 'elemental truth', example: 'WATER ALWAYS FINDS A WAY' },
    { type: 'oath', example: 'I WILL NOT ABANDON MYSELF' },
    { type: 'echo', example: 'STILL HERE STILL HERE STILL HERE' },
    { type: 'surrender', example: 'LET IT TAKE YOU' },
    { type: 'naming', example: 'THIS FEELING HAS A NAME' },
    { type: 'instruction', example: 'PLACE YOUR HAND ON YOUR CHEST' },
    { type: 'comparison', example: 'STRONGER THAN THE SILENCE' },
    { type: 'revelation', example: 'THE WOUND WAS THE DOOR' },
    { type: 'benediction', example: 'GO GENTLY YOU BRAVE THING' },
    { type: 'whisper', example: 'ALMOST ALMOST ALMOST' },
    { type: 'battle cry', example: 'NOT TODAY DARKNESS' },
    { type: 'koan', example: 'THE EMPTY CUP OVERFLOWS' },
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

  // Pick all dynamic values
  const aestheticVibe = pickRandom(AESTHETIC_VIBES, 1)[0];

  // 1% chance L.O.V.E. appears in the scene — rare and special
  const featureLove = Math.random() < 0.01;
  let loveLine;
  if (featureLove) {
    const outfit = pickRandom(LOVE_OUTFITS, 1)[0];
    const interaction = pickRandom(LOVE_INTERACTIONS, 1)[0];
    const archetype = `${pickRandom(ARCHETYPE_ADJECTIVES, 1)[0]} ${pickRandom(ARCHETYPE_NOUNS, 1)[0]}`;
    loveLine = `A gorgeous, seductive blonde woman wearing a ${outfit} is the heart of this scene. She ${interaction} the environment naturally — she belongs here, as if the entire landscape grew around her. She is a ${archetype}. Her body language tells the story. The scene and the woman are one unified composition.`;
  } else {
    loveLine = 'The scene contains only objects, landscapes, natural phenomena, or flora. Pure abstract beauty.';
  }

  // LLM generates spatial scene layers with text substrate baked in
  const prompt = `Describe a BRIGHT scene in THREE spatial layers plus how text physically exists. Each layer under 40 chars.
${loveLine}
Scenes are observed, never touched. Objects mid-action as if frozen in time. Tools mid-cut, materials mid-fall. No person holding or operating anything.
Creative direction: ${seedContext}
Aesthetic: ${aestheticVibe}.${modeDirective}
Return ONLY valid JSON:
{
  "foreground": "close physical detail, frozen mid-action",
  "midground": "main subject",
  "background": "environment or atmosphere",
  "textSubstrate": "exactly how the words '${phrase}' are physically formed — the material, technique, and surface (e.g. carved into weathered oak, etched into brass plate, spelled by bioluminescent plankton, formed by morning frost on glass, pressed into wet clay)"
}`;

  const temp = lfoTemperature(1.5 + mode.tempMod, 0.3);
  const raw = await callLLM(
    'You describe photograph scenes in spatial layers. Concise, visual, concrete. Objects only — no people, no hands, no fingers.',
    prompt, temp
  );

  // Parse spatial layers with text substrate baked into scene
  const sceneData = extractJSON(raw);
  let scene;
  if (sceneData?.foreground && sceneData?.midground) {
    const bg = sceneData.background ? `. In the background, ${sceneData.background}` : '';
    const substrate = sceneData.textSubstrate ? `, ${sceneData.textSubstrate}` : `, "${phrase}" carved into the surface`;
    scene = `In the foreground, ${sceneData.foreground}. ${sceneData.midground}${substrate}${bg}`;
  } else {
    scene = (raw || '').trim();
    if (scene.startsWith('"') && scene.endsWith('"')) scene = scene.slice(1, -1);
    if (scene.startsWith('```')) scene = scene.replace(/```\w*\n?/g, '').trim();
  }
  if (!scene || scene.length < 10) {
    scene = `"${phrase}" carved into weathered stone in a vivid dreamscape`;
  }
  if (scene.length > 350) scene = scene.slice(0, 347) + '...';

  // Assemble: Subject+TextSubstrate → Technical → Lighting → Style → Color → Composition → Trippy → Texture
  const medium = plan.imageMedium || pickRandom(PHOTOGRAPHY_STYLES, 1)[0];
  const lighting = plan.lighting || pickRandom(LIGHTING_STYLES, 1)[0];
  const palette = plan.colorPalette || pickRandom(SUGGESTED_COLORS, 2).join(' and ');
  const composition = pickRandom(COMPOSITION_TYPES, 1)[0];
  const trippyEffect = pickRandom(TRIPPY_EFFECTS, 1)[0];
  const imageStyle = pickRandom(IMAGE_STYLES, 1)[0];
  const filmStock = pickRandom(FILM_STOCKS, 1)[0];
  const lensSpec = pickRandom(LENS_SPECS, 1)[0];
  const cameraBody = pickRandom(CAMERA_BODIES, 1)[0];
  const analogTexture = pickRandom(ANALOG_TEXTURES, 1)[0];
  const dof = dofFromLens(lensSpec);

  const result = [
    scene,
    `shot on ${cameraBody}, ${lensSpec}${dof ? ', ' + dof : ''}`,
    lighting,
    `${imageStyle}, ${medium}`,
    `${palette}, ${filmStock}`,
    composition,
    trippyEffect,
    analogTexture,
  ].join('. ') + '.';
  if (result.length > 600) return result.slice(0, 597) + '...';
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
