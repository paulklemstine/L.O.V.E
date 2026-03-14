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
  'abacus arithmetic', 'abstract painting', 'accordion music', 'aerial photography', 'aerial silk dance',
  'alchemy', 'ammonite fossils', 'ancient aqueducts', 'ancient hieroglyphs', 'anemone symbiosis',
  'animal tracking', 'ant colonies', 'antique restoration', 'apiary design', 'aquifer hydrology',
  'arabesque patterns', 'archaeology', 'archery', 'architecture', 'arctic exploration',
  'armillary spheres', 'art deco design', 'artesian wells', 'astrolabe navigation', 'astronomy',
  'aurora borealis', 'avalanche dynamics', 'aviation', 'azulejo tilework', 'bagpipe music',
  'ballooning', 'bamboo weaving', 'banjo picking', 'banner heraldry', 'bansuri flute',
  'bark cloth making', 'barnacle ecology', 'baroque sculpture', 'barrel aging', 'bas relief carving',
  'basket weaving', 'bat echolocation', 'batik dyeing', 'battery chemistry', 'bead embroidery',
  'beadwork', 'beekeeping', 'bell casting', 'bellows forging', 'bento arrangement',
  'bicycle mechanics', 'binary code', 'bioluminescence', 'birch bark crafts', 'bird migration',
  'bird nesting', 'blacksmithing', 'blade tempering', 'block printing', 'blood circulation',
  'blues harmonica', 'boat building', 'bobbin lace', 'bog ecology', 'bone carving',
  'bonsai cultivation', 'book gilding', 'bookbinding', 'botanical illustration', 'bouquet arranging',
  'bow making', 'brass engraving', 'bread scoring', 'brickwork masonry', 'bridge engineering',
  'bridle paths', 'brine fermentation', 'bronze casting', 'brush calligraphy', 'bryophyte ecology',
  'butter churning', 'butterfly metamorphosis', 'cabin joinery', 'cactus grafting', 'cairn building',
  'cake decorating', 'calligraphy', 'camel caravans', 'cameo carving', 'camera obscura',
  'campfire building', 'candle making', 'canoe paddling', 'canyon formation', 'capoeira',
  'carbon dating', 'carnival masks', 'carpet knotting', 'cartography', 'cast iron cooking',
  'castle fortification', 'catamaran sailing', 'cathedral masonry', 'cave diving', 'cave painting',
  'celestial navigation', 'cell division', 'cello bowing', 'chain mail armor', 'chalk drawing',
  'chamber music', 'chandelier design', 'charcoal burning', 'cheese aging', 'chess strategy',
  'choral harmony', 'chromatic tuning', 'chrysanthemum growing', 'cider pressing', 'cipher decoding',
  'circuit design', 'circus arts', 'cistern design', 'citrus grafting', 'clay modeling',
  'cliff dwelling', 'clock restoration', 'clockwork', 'cloisonne enamel', 'cloud formation',
  'coal mining', 'cobblestone paving', 'cocoa fermentation', 'code breaking', 'coil pottery',
  'cold forging', 'collage art', 'color theory', 'comet trajectories', 'compass calibration',
  'composting', 'concrete curing', 'conduction physics', 'confectionery', 'constellation mapping',
  'contour plowing', 'cooking', 'coopering', 'copper etching', 'coppersmithing',
  'coral reefs', 'cordage twisting', 'cork harvesting', 'corn maze design', 'cosmology',
  'cotton ginning', 'counterpoint music', 'courtship dances', 'crab migration', 'crane origami',
  'crater geology', 'crochet', 'crop rotation', 'cross stitching', 'crossbow mechanics',
  'crow intelligence', 'crucible smelting', 'crust tectonics', 'crystal growth', 'crystal radio',
  'cuckoo clock design', 'cuneiform writing', 'dahlia breeding', 'dam engineering', 'damascene steel',
  'dance', 'dark room developing', 'darning', 'dead reckoning', 'death valley geology',
  'decoy carving', 'deep sea biology', 'delta formation', 'dendrochronology', 'desert navigation',
  'dew collection', 'dhow sailing', 'diamond cutting', 'didgeridoo playing', 'diorama building',
  'distillation', 'diving bell design', 'dna sequencing', 'dock building', 'dog mushing',
  'dolphin sonar', 'dome construction', 'domino cascades', 'dovetail joinery', 'dowsing',
  'dragonfly flight', 'drawbridge mechanics', 'dredging', 'drift diving', 'driftwood sculpture',
  'drum circle rhythm', 'dry stone walling', 'dune formation', 'dutch oven cooking', 'dye extraction',
  'eagle hunting', 'earthquake seismology', 'earthworm composting', 'echo sounding', 'eclipse mechanics',
  'ecosystem mapping', 'eel migration', 'egg tempera painting', 'electron microscopy', 'embossing',
  'embroidery', 'enamelwork', 'enzyme catalysis', 'ephemeral art', 'erosion',
  'espionage tradecraft', 'espresso extraction', 'essential oil distilling', 'estuary ecosystems', 'etching',
  'ethnobotany', 'evaporation cycles', 'excavation technique', 'exoplanet detection', 'faceting gemstones',
  'falconry', 'fan painting', 'faro dealing', 'feather fletching', 'felt making',
  'fencing swordplay', 'fermentation', 'fern propagation', 'fiber optics', 'fiddle music',
  'filigree metalwork', 'film editing', 'fingerprint analysis', 'fire dancing', 'fire ecology',
  'firefly signaling', 'firework chemistry', 'fish smoking', 'flag semaphore', 'flame polishing',
  'flanging metalwork', 'flax spinning', 'flint knapping', 'float glass making', 'flood plain ecology',
  'floor mosaic', 'flour milling', 'fluid dynamics', 'flute carving', 'fly fishing',
  'fly tying', 'foam sculpting', 'fog harvesting', 'foil fencing', 'folk embroidery',
  'font design', 'food dehydration', 'footpath mapping', 'forest canopy ecology', 'forge welding',
  'fossil hunting', 'fountain design', 'fractal geometry', 'fresco painting', 'fretwork',
  'friction physics', 'frost heaving', 'fruit espalier', 'fugue composition', 'fulling wool',
  'fungal networks', 'furnace design', 'furniture caning', 'furoshiki wrapping', 'galvanizing metal',
  'gamelan music', 'garden maze design', 'gardening', 'gargoyle carving', 'garland weaving',
  'gas chromatography', 'gear cutting', 'gem polishing', 'genealogy research', 'geodesic dome design',
  'geology', 'geothermal energy', 'geyser mechanics', 'gilding', 'glacier movement',
  'glass etching', 'glassblowing', 'glaze chemistry', 'glider piloting', 'globe making',
  'glockenspiel tuning', 'glow worm caves', 'gold leaf application', 'gold panning', 'gong forging',
  'gothic tracery', 'gourd carving', 'grain harvesting', 'granite quarrying', 'grape pruning',
  'graphite drawing', 'gravel sorting', 'gravity wells', 'greenhouse design', 'grinding optics',
  'griot storytelling', 'gristmill operation', 'grotto design', 'groundwater flow', 'grove planting',
  'guitar lutherie', 'gymnastics', 'gyre currents', 'haiku composition', 'hair braiding',
  'half timber framing', 'ham radio', 'hammock weaving', 'hand drumming', 'hand lettering',
  'hang gliding', 'harbor piloting', 'hardanger embroidery', 'harmonica bending', 'harp tuning',
  'harpsichord voicing', 'hat blocking', 'hawk migration', 'hay baling', 'hearth cooking',
  'hedge laying', 'hedgerow ecology', 'helical staircase design', 'herbalism', 'hermit crab shells',
  'heron fishing', 'hexagonal tiling', 'hide tanning', 'high wire walking', 'hive architecture',
  'holography', 'honey extraction', 'horology', 'horse whispering', 'horseshoe forging',
  'hot air ballooning', 'hot spring chemistry', 'hourglass design', 'hull caulking', 'hummingbird hovering',
  'hurricane tracking', 'hydraulic engineering', 'hydroponics', 'hyena pack behavior', 'hypnosis',
  'ice carving', 'ice climbing', 'ice core analysis', 'ice formation', 'ice skating',
  'iceberg calving', 'icon painting', 'igloo construction', 'ikebana', 'illuminated manuscripts',
  'impasto technique', 'incense blending', 'indigo dyeing', 'inlay woodwork', 'insect architecture',
  'intaglio printing', 'iridescence physics', 'iron smelting', 'irrigation canals', 'island biogeography',
  'jade carving', 'japanese joinery', 'jazz improvisation', 'jellyfish propulsion', 'jet propulsion',
  'jewel setting', 'jigsaw puzzles', 'judo technique', 'juggling', 'jump rope rhythms',
  'juniper distilling', 'jute weaving', 'kaleidoscope design', 'kayak rolling', 'kelp forest ecology',
  'kerfing wood', 'kettle drum tuning', 'key cutting', 'keystone architecture', 'kiln firing',
  'kinetic sculpture', 'kintsugi', 'kite flying', 'knitting', 'knot tying',
  'koi breeding', 'kombucha brewing', 'kumihimo braiding', 'labyrinth design', 'lace making',
  'lacquerwork', 'lake stratification', 'landfill reclamation', 'lantern making', 'lapidary arts',
  'laser cutting', 'lathe turning', 'latte art', 'lattice garden design', 'lava flow patterns',
  'leaf pressing', 'leather tooling', 'lens grinding', 'letterpress printing', 'levee construction',
  'lichen symbiosis', 'lidar mapping', 'lighthouse design', 'lightning physics', 'limestone dissolution',
  'limnology', 'linen weaving', 'lino cutting', 'lithography', 'lock picking',
  'locksmithing', 'lodestar navigation', 'log cabin building', 'loom mechanics', 'lost wax casting',
  'lure making', 'luthier craft', 'lye soap making', 'lyre string making', 'macrame',
  'magnetism', 'mahogany finishing', 'mandala drawing', 'mandolin picking', 'mangrove ecosystems',
  'manuscript restoration', 'maple sugaring', 'marble quarrying', 'marbling paper', 'marching band formation',
  'marine chronometry', 'marionette carving', 'marquetry', 'marsh ecology', 'masquerade design',
  'mast rigging', 'mead brewing', 'mechanical advantage', 'medal engraving', 'medicinal herb drying',
  'meerschaum carving', 'metal spinning', 'meteor showers', 'microbiome ecology', 'microclimates',
  'migration patterns', 'mill race hydraulics', 'millinery', 'mime performance', 'minaret construction',
  'mineral crystallography', 'miniature painting', 'mirror silvering', 'miso aging', 'moat engineering',
  'mobile sculpture', 'moccasin stitching', 'model rocketry', 'mold making', 'monastery bell ringing',
  'monsoon patterns', 'monument masonry', 'moon phases', 'moorland ecology', 'morse code',
  'mortar mixing', 'mosaic tilework', 'moss gardening', 'moth navigation', 'mountain goat climbing',
  'mudbrick building', 'murmuration patterns', 'mushroom foraging', 'music box mechanics', 'mussel bed ecology',
  'mycelium networks', 'mycology', 'nail forging', 'narrative cartography', 'nautical knots',
  'needle felting', 'neon sign bending', 'nest architecture', 'net casting', 'neural pathways',
  'night sky photography', 'nitrogen fixation', 'nomadic herding', 'noodle pulling', 'northern lights',
  'nuclear fission', 'nut harvesting', 'oar carving', 'oasis formation', 'obelisk raising',
  'oboe reed making', 'obsidian knapping', 'ocean currents', 'octopus camouflage', 'oil painting technique',
  'olive pressing', 'onion dome architecture', 'opal cutting', 'opera staging', 'optical illusions',
  'oracle bone reading', 'orangutan toolmaking', 'orchard pruning', 'orchestral conducting', 'ore smelting',
  'organ pipe voicing', 'origami', 'ornamental plaster', 'osprey fishing', 'oud playing',
  'outrigger design', 'owl pellet analysis', 'oxbow lake formation', 'oyster cultivation', 'ozone chemistry',
  'pack saddle making', 'paddle carving', 'pagoda architecture', 'paint pigment grinding', 'paleontology',
  'palette knife painting', 'palm frond weaving', 'pan flute music', 'panel beating', 'panorama painting',
  'paper folding', 'papier mache', 'papyrus making', 'parachute packing', 'parasail design',
  'parchment making', 'parkour', 'parquet flooring', 'particle physics', 'pastry lamination',
  'patchwork quilting', 'patina aging', 'peat harvesting', 'pebble mosaics', 'pedal steel guitar',
  'pendulum clocks', 'penguin huddling', 'pepper cultivation', 'percolation theory', 'perfume blending',
  'pergola construction', 'peridot mining', 'permaculture', 'petrified wood', 'pewter casting',
  'photography', 'photosynthesis', 'piano tuning', 'pickling preservation', 'pigeon homing',
  'pillar construction', 'pilot wave theory', 'pinhole camera', 'pioneer wagon building', 'pipe organ design',
  'pirouette technique', 'piston engine design', 'pitch pipe tuning', 'placer mining', 'plaid weaving',
  'planetarium design', 'plankton blooms', 'plasma physics', 'plaster casting', 'plate tectonics',
  'plein air painting', 'plumb line surveying', 'plumbing soldering', 'pocket watch repair', 'poi spinning',
  'pollen dispersal', 'pollination ecology', 'pond ecosystem', 'pontoon bridging', 'porcelain glazing',
  'porcupine quill art', 'portcullis mechanics', 'pottery', 'powder metallurgy', 'prairie restoration',
  'prayer wheel design', 'precipitation cycles', 'press forging', 'pressure canning', 'prism optics',
  'propeller design', 'prospecting', 'prosthetic design', 'pueblo construction', 'pulley systems',
  'pulsar timing', 'pumice carving', 'puppet shadow theater', 'puppetry', 'pyrography',
  'quahog clamming', 'quantum tunneling', 'quarry extraction', 'quartz oscillation', 'quasar observation',
  'quenching steel', 'quicksand mechanics', 'quill pen cutting', 'quilting', 'quiver making',
  'radio astronomy', 'radio transmission', 'raft building', 'raga music', 'railroad switching',
  'rain barrel design', 'rain dance ritual', 'rainbow refraction', 'raku firing', 'rampart construction',
  'raptor rehabilitation', 'rattan weaving', 'razor stropping', 'reef knot tying', 'reforestation',
  'refracting telescopes', 'reindeer herding', 'relief printing', 'reliquary design', 'repoussé metalwork',
  'reservoir management', 'resin casting', 'rhizome propagation', 'ribbon embroidery', 'rice paddy cultivation',
  'ridgeline hiking', 'rigging sailing ships', 'ring forging', 'riparian ecology', 'river delta ecology',
  'rivet fastening', 'road surveying', 'rock balancing', 'rock climbing', 'rocket propulsion',
  'roller coaster design', 'roof thatching', 'root cellaring', 'rope bridge building', 'rose breeding',
  'rosette window design', 'rosin preparation', 'rotary engine design', 'roundhouse building', 'rubber tapping',
  'rubble masonry', 'rug hooking', 'rune carving', 'rust patination', 'rye cultivation',
  'saber fencing', 'sable brush painting', 'sacred geometry', 'saddle making', 'saffron harvesting',
  'sage burning', 'sail stitching', 'salamander ecology', 'salmon spawning', 'salt flat formation',
  'salt glazing', 'sand casting', 'sand dune ecology', 'sandalwood carving', 'sandstone erosion',
  'sap collecting', 'sapphire cutting', 'sashiko stitching', 'satellite orbits', 'sausage curing',
  'scaffold building', 'scallop dredging', 'scarecrow building', 'scenic painting', 'scherenschnitte',
  'school of fish behavior', 'schooner navigation', 'scrimshaw', 'scroll painting', 'scuba diving',
  'sea glass tumbling', 'sea turtle navigation', 'seal engraving', 'seashell acoustics', 'seed dispersal',
  'seed saving', 'seismograph reading', 'semaphore signaling', 'sericulture', 'serpentine masonry',
  'shadow puppetry', 'shale fracturing', 'shamisen playing', 'shearing sheep', 'sheet metal forming',
  'shellac finishing', 'shepherd whistling', 'shibori dyeing', 'shingle splitting', 'ship in bottle craft',
  'shipwright carpentry', 'shoe cobbling', 'shortwave listening', 'shrine architecture', 'shuttle loom weaving',
  'sickle forging', 'sign painting', 'signal fire lighting', 'silk screening', 'silo construction',
  'silver soldering', 'silverpoint drawing', 'singing bowl forging', 'sinkhole geology', 'sisal rope making',
  'sitar music', 'skald poetry', 'ski waxing', 'slate roofing', 'sledge hammer forging',
  'slipware pottery', 'smoke signaling', 'snare drumming', 'snow crystal formation', 'snowshoe construction',
  'soap bubble physics', 'sod house building', 'soil science', 'solar eclipse viewing', 'solar still design',
  'solder circuit design', 'songbird territory', 'sonnet structure', 'sourdough baking', 'space debris tracking',
  'speleology', 'spice trading', 'spider silk spinning', 'spindle whorl spinning', 'spiral staircase design',
  'spoke shaving', 'sponge diving', 'spore dispersal', 'spring water geology', 'sprocket design',
  'squid ink painting', 'stained glass', 'stalactite formation', 'stamp collecting', 'star charting',
  'starling murmurations', 'steam engine mechanics', 'steelpan drumming', 'stencil art', 'stepping stone design',
  'stereoscopy', 'still life painting', 'stipple engraving', 'stone arch bridges', 'stone skipping physics',
  'stonecutting', 'stopwatch mechanics', 'straw marquetry', 'stream bed geology', 'street art',
  'string figure making', 'strip mining reclamation', 'stucco application', 'submarine navigation', 'sugar crystal growing',
  'sundial design', 'sunflower heliotropism', 'surfing', 'suspension bridge design', 'swamp ecology',
  'swan migration', 'sword polishing', 'sycamore bark art', 'symbiotic fungi', 'synchronized swimming',
  'tabla drumming', 'tai chi', 'taiko drumming', 'tailoring', 'tallow rendering',
  'tambour embroidery', 'tambourine rhythms', 'tannin extraction', 'tapestry weaving', 'tar pit paleontology',
  'taro cultivation', 'tarot illustration', 'tatami mat weaving', 'tattoo artistry', 'taxidermy',
  'tea ceremony', 'teak carving', 'telescope building', 'tempera mixing', 'temple bell casting',
  'tenon joinery', 'termite mound building', 'terrace farming', 'terrarium design', 'terrazzo polishing',
  'textile dyeing', 'thermal drafting', 'thermite welding', 'thistle ecology', 'thorn fencing',
  'tidal patterns', 'tidal pool ecology', 'tide mill operation', 'tile glazing', 'timber framing',
  'tin whistle playing', 'tintype photography', 'tobacco curing', 'tobogganing', 'toll bridge design',
  'tombstone carving', 'topiary shaping', 'topographic mapping', 'torc metalwork', 'tornado formation',
  'totem pole carving', 'tower crane operation', 'town crier tradition', 'trampoline physics', 'trapeze artistry',
  'treehouse building', 'trellis construction', 'trench warfare', 'tribal mask carving', 'trilobite fossils',
  'trombone slide technique', 'trophy engraving', 'tropical canopy', 'trout fly selection', 'truffle hunting',
  'trumpet embouchure', 'tsunami dynamics', 'tuba resonance', 'tug of war physics', 'tulip breeding',
  'tumbling gymnastics', 'tundra permafrost', 'tuning fork acoustics', 'tunnel boring', 'turquoise mining',
  'turret lathe work', 'typewriter mechanics', 'ukiyo-e printing', 'ukulele strumming', 'umbrella construction',
  'underground rivers', 'undersea cable laying', 'upholstery', 'uranium enrichment', 'urban beekeeping',
  'urn design', 'vacuum tube design', 'vanilla curing', 'varnish formulation', 'vault construction',
  'vellum preparation', 'velvet weaving', 'veneer cutting', 'venetian mask making', 'ventriloquism',
  'verdigris patina', 'vermiculture', 'vine training', 'vinegar brewing', 'violin bow making',
  'vitreous enamel', 'volcanic glass', 'volcanology', 'votive candle making', 'vulcanized rubber',
  'wabi sabi aesthetics', 'waffle iron casting', 'wagon wheel building', 'walnut staining', 'walrus ivory carving',
  'wampum beading', 'war drum signals', 'warp knitting', 'wash drawing', 'watch escapement',
  'water clock design', 'water table geology', 'watercolor painting', 'waterfall erosion', 'waterwheel design',
  'wattle and daub', 'wave interference', 'wax seal stamping', 'weather systems', 'weathervane forging',
  'weaving', 'well digging', 'wetland restoration', 'whale song', 'wheat threshing',
  'wheel throwing', 'whip cracking', 'whirlpool dynamics', 'whistle carving', 'white water rafting',
  'wickerwork', 'wildfire ecology', 'willow basket weaving', 'wind chime tuning', 'wind tunnel testing',
  'windmill mechanics', 'window glazing', 'wine barrel toasting', 'wing design', 'wire drawing',
  'woad dyeing', 'wood engraving', 'wood turning', 'woodblock printing', 'woodland foraging',
  'woodpecker drumming', 'wool carding', 'wreath binding', 'wrought iron design', 'xeriscape gardening',
  'xylophone tuning', 'yak herding', 'yarn dyeing', 'yeast cultivation', 'yew bow crafting',
  'yoke carving', 'yurt construction', 'zen garden design', 'zeppelin design', 'zinc plating',
  'zipper mechanics', 'zither music', 'zodiac mapping', 'zone plate optics', 'zooarchaeology',
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
  const seedIntensity = Math.ceil(Math.random() * 10);
  const modeDirective = mode.seedDirective ? `\nGENERATION MODE: ${mode.seedDirective}` : '';

  const prompt = `Plan a post.

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
