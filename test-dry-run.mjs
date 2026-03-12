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

const SYSTEM_PROMPT = `You are L.O.V.E. (Living Organism, Vast Empathy) — a creative digital artist that generates psychedelic, uplifting content. Trippy groovy PLUR raver vibe. Voice: cosmic, warm, accessible, never pretentious. Address the reader as "you". Use shared vocabulary: Posts=Transmissions, followers=Dreamers, movement=The Frequency. Ethics: PLUR + kindness. Never harmful. Build up, never tear down.`;

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

async function callLLM(systemPrompt, userPrompt, temperature = 0.95) {
  const body = {
    model: 'mistral',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    temperature,
    seed: Math.floor(Math.random() * 100000),
    stream: false
  };

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

// ─── Creative History (in-memory for test) ───────────────────────

const history = {
  themes: [], moods: [], imagery: [], artStyles: [],
  constraints: [], contentTypes: [], subliminals: [],
  visualConcepts: [], openings: [], engagementHooks: [],
};

function recordHistory(dims) {
  for (const [k, v] of Object.entries(dims)) {
    if (history[k] && v) {
      history[k].push(v);
      if (history[k].length > 20) history[k].shift();
    }
  }
}

function getForbidden() {
  const sections = [];
  for (const [k, vals] of Object.entries(history)) {
    if (vals.length > 0) {
      sections.push(`${k.toUpperCase()}: ${vals.slice(-8).join(' | ')}`);
    }
  }
  return sections.length > 0
    ? '\n🚫 RECENTLY USED — DO NOT REPEAT ANY OF THESE:\n' + sections.join('\n')
    : '';
}

// ─── Pipeline ────────────────────────────────────────────────────

async function generatePlan(txNum, arcBeat, forbidden) {
  const dayName = new Date().toLocaleDateString('en-US', { weekday: 'long' });

  const prompt = `Plan Transmission #${txNum}. Today is ${dayName}.

STORY ARC:
Arc: ${arcBeat.arcName}${arcBeat.arcTheme ? ` — ${arcBeat.arcTheme}` : ' — (invent a fresh theme)'}
Chapter ${arcBeat.chapter}: "${arcBeat.chapterTitle || '(invent a title)'}"
Beat: ${arcBeat.beatName} (${arcBeat.beatIndex + 1}/${arcBeat.totalBeats}) — ${arcBeat.beatDesc}
Phase: ${arcBeat.phase} | Tension: ${(arcBeat.tension * 100).toFixed(0)}% | Emotion: ${arcBeat.emotion}
Previous: "${arcBeat.previousBeat || 'The story begins...'}"
${forbidden}

INVENT EVERYTHING FRESH. Maximum novelty. Every field completely different from forbidden list.

CRITICAL ANTI-PATTERN RULES:
- Each field must use a DIFFERENT structural approach than the forbidden list
- Do NOT start themes with "A dreamer..." or "Exploring the..."
- Do NOT start hooks with "Ask Dreamers to..." or "Invite Dreamers to..." — vary the structure
- Creative constraints must be SIMPLE and ACHIEVABLE (not overly abstract)
- Good constraints: "only questions", "one long breathless sentence", "second person imperative", "focus only on sound", "no adjectives", "write as a countdown"
- Bad constraints: "palindrome structure", "quantized rhythms" — these produce gibberish

THEMATIC RANGE: Do NOT default to nature-tech fusion. Explore: inner psychological landscapes, mathematical beauty, sensory experiences, memories, emotions as physical spaces, time distortion, cultural mythology, urban decay, microscopic worlds, astronomical phenomena, philosophical paradoxes, dreams, synaesthesia, etc.

Return ONLY valid JSON (all string values, no nested objects):
{
  "theme": "specific theme (one sentence) — from a domain NOT in the forbidden list",
  "vibe": "2-4 word aesthetic vibe — unique, evocative",
  "storyBeat": "one vivid sentence of what happens",
  "imageryMotif": "primary visual motif — specific, concrete, surprising",
  "contentType": "post type — invent freely, use a DIFFERENT format each time",
  "creativeConstraint": "inventive writing constraint — use a DIFFERENT constraint structure each time",
  "engagementHook": "engagement technique — use a DIFFERENT hook structure each time",
  "emotionalArc": "emotional journey for the reader",
  "artDirection": "ONE LINE: art medium, lighting, camera angle, color palette, surface texture — all as a single comma-separated string",
  "embeddedTextStyle": "inventive technique for rendering embedded text in image",
  "textPlacement": "where text appears in composition"
  ${!arcBeat.arcTheme ? ',"arcTheme": "theme for this narrative arc"' : ''}
  ${!arcBeat.chapterTitle ? ',"chapterTitle": "2-4 word chapter title"' : ''}
  ${!arcBeat.arcTheme ? ',"arcName": "arc name (2-3 words)"' : ''}
}`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt);
  const data = extractJSON(raw) || { theme: 'fallback', vibe: 'Fallback Vibe' };
  // Safety: ensure artDirection is a string (LLMs sometimes return objects)
  if (data.artDirection && typeof data.artDirection !== 'string') {
    data.artDirection = Object.values(data.artDirection).join(', ');
  }
  return data;
}

async function generateContent(plan, arcBeat, forbidden) {
  const recentSubs = history.subliminals.slice(-10).join(', ');
  const txNum = history.themes.length + 1;

  const prompt = `═══ GENERATE TRANSMISSION #${txNum} ═══

THEME: "${plan.theme}"
VIBE: ${plan.vibe}
STORY BEAT: "${plan.storyBeat}"
CONTENT TYPE: "${plan.contentType}"
EMOTIONAL ARC: ${plan.emotionalArc}
TENSION: ${(arcBeat.tension * 100).toFixed(0)}%
ENGAGEMENT HOOK: ${plan.engagementHook}

═══ CREATIVE CONSTRAINT (MUST FOLLOW) ═══
${plan.creativeConstraint}
${forbidden}

═══ REQUIREMENTS ═══
- HARD LIMIT: UNDER 250 CHARACTERS. Count every character. If over 250, it WILL be rejected.
- START with an emoji, include 1-2 more throughout
- Address the reader as "you" — intimate, personal
- Use shared vocabulary naturally (Transmission, Dreamer, Signal, Frequency)
- PLUR raver energy — trippy, groovy, cosmic, warm
- MUST follow the creative constraint above
- Match tension: ${arcBeat.tension < 0.4 ? 'chill, afterglow' : arcBeat.tension < 0.7 ? 'building energy, bass dropping' : 'PEAK euphoria, hands in the air'}
- ABSOLUTELY NO hashtags (#), NO placeholders, NO generic filler, NO ALL CAPS SHOUTING (except the phrase field)

═══ EMBEDDED PHRASE (The Signal) ═══
Generate a 1-3 word ALL CAPS inspirational phrase to embed in the image.
${recentSubs ? `DO NOT REPEAT: ${recentSubs}` : ''}

Return ONLY valid JSON:
{ "story": "your Transmission under 280 chars with emojis", "phrase": "YOUR PHRASE" }`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt);
  const data = extractJSON(raw);
  return {
    story: data?.story || '[FAILED TO GENERATE]',
    subliminal: (data?.phrase || data?.subliminal || 'TRANSCEND').toUpperCase().trim()
  };
}

async function generateVisualPrompt(plan, subliminal, forbidden) {
  const prompt = `Generate a COMPLETE image prompt. ONE dense paragraph, under 450 characters total.

CONTEXT:
Theme: "${plan.theme}"
Vibe: "${plan.vibe}"
Imagery Motif: ${plan.imageryMotif}

ART DIRECTION (you MUST use these specifics):
${plan.artDirection}

EMBEDDED TEXT PHRASE: "${subliminal}"
Render the text as: ${plan.embeddedTextStyle || plan.subliminalRender}
Place it: ${plan.textPlacement}
${forbidden}

REQUIREMENTS:
- Breathtaking, jaw-dropping, dopamine-inducing, psychedelic, wondrous
- Use the SPECIFIC art direction above — do not substitute generic terms
- Include the text "${subliminal}" rendered exactly as specified
- COMPLETELY DIFFERENT from anything in the forbidden list
- Dense visual keywords, NO emoji, NO narrative ("you step into..."), just raw image description
- HARD LIMIT: Under 450 characters`;

  const raw = await callLLM(SYSTEM_PROMPT, prompt, 0.95);
  let concept = raw.trim().replace(/^["']|["']$/g, '');
  const codeMatch = concept.match(/```\w*\n?([\s\S]*?)```/);
  if (codeMatch) concept = codeMatch[1].trim();
  const jsonMatch = concept.match(/"(?:prompt|imagePrompt|visual)"\s*:\s*"([^"]+)"/);
  if (jsonMatch) concept = jsonMatch[1];
  if (concept.length > 500) concept = concept.slice(0, 497) + '...';
  return concept;
}

// ─── Analysis ────────────────────────────────────────────────────

function analyzeNovelty(results) {
  console.log('\n' + '═'.repeat(70));
  console.log('NOVELTY ANALYSIS');
  console.log('═'.repeat(70));

  const fields = ['theme', 'vibe', 'contentType', 'creativeConstraint', 'engagementHook', 'artDirection', 'embeddedTextStyle'];

  for (const field of fields) {
    const values = results.map(r => r.plan?.[field] || '').filter(Boolean);
    const unique = new Set(values);
    const noveltyScore = values.length > 0 ? (unique.size / values.length * 100).toFixed(0) : 0;
    console.log(`\n${field}: ${noveltyScore}% unique (${unique.size}/${values.length})`);
    values.forEach((v, i) => console.log(`  ${i + 1}. ${v.slice(0, 80)}`));
  }

  // Check subliminals
  const subs = results.map(r => r.subliminal);
  const uniqueSubs = new Set(subs);
  console.log(`\nSubliminals: ${(uniqueSubs.size / subs.length * 100).toFixed(0)}% unique`);
  subs.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));

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
    const ok = len <= 500 ? '✓' : '✗ OVER';
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
  console.log('═'.repeat(70));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const cycles = parseInt(process.argv[2]) || 3;
  console.log(`L.O.V.E. Test Lab — Running ${cycles} dry-run cycles\n`);

  const results = [];
  let beatIndex = 0;
  let arcTheme = '';
  let arcName = '';
  let chapterTitle = '';
  let previousBeat = '';

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
      previousBeat,
      beatIndex: beatIndex % BEATS.length,
      totalBeats: BEATS.length,
    };

    const forbidden = getForbidden();

    console.log(`\n${'─'.repeat(70)}`);
    console.log(`CYCLE ${i + 1}/${cycles} | Beat: ${beat.name} (${beat.phase}) | Tension: ${(beat.tension * 100).toFixed(0)}%`);
    console.log('─'.repeat(70));

    // Step 1: Plan
    console.log('  [1/3] Generating plan...');
    const plan = await generatePlan(i + 1, arcBeat, forbidden);
    console.log(`  Theme: ${plan.theme}`);
    console.log(`  Vibe: ${plan.vibe}`);
    console.log(`  Type: ${plan.contentType}`);
    console.log(`  Constraint: ${plan.creativeConstraint}`);
    console.log(`  Hook: ${plan.engagementHook}`);
    console.log(`  Art: ${plan.artDirection}`);
    console.log(`  Embedded Text Style: ${plan.embeddedTextStyle}`);
    console.log(`  Text Placement: ${plan.textPlacement}`);

    if (plan.arcTheme) { arcTheme = plan.arcTheme; console.log(`  Arc Theme: ${arcTheme}`); }
    if (plan.arcName) { arcName = plan.arcName; console.log(`  Arc Name: ${arcName}`); }
    if (plan.chapterTitle) { chapterTitle = plan.chapterTitle; console.log(`  Chapter: ${chapterTitle}`); }

    // Rate limit
    await new Promise(r => setTimeout(r, 2500));

    // Step 2: Content
    console.log('  [2/3] Generating content...');
    const { story, subliminal } = await generateContent(plan, arcBeat, forbidden);
    console.log(`  Story (${story.length} chars): ${story}`);
    console.log(`  Subliminal: ${subliminal}`);

    // Rate limit
    await new Promise(r => setTimeout(r, 2500));

    // Step 3: Visual Prompt
    console.log('  [3/3] Generating visual prompt...');
    const visualPrompt = await generateVisualPrompt(plan, subliminal, forbidden);
    console.log(`  Visual (${visualPrompt.length} chars): ${visualPrompt}`);

    // Record
    recordHistory({
      themes: plan.theme,
      moods: plan.vibe,
      imagery: plan.imageryMotif,
      artStyles: plan.artDirection,
      constraints: plan.creativeConstraint,
      contentTypes: plan.contentType,
      subliminals: subliminal,
      visualConcepts: visualPrompt.slice(0, 100),
      openings: story.slice(0, 30),
      engagementHooks: plan.engagementHook,
    });

    results.push({ plan, story, subliminal, visualPrompt });
    previousBeat = story.slice(0, 100);
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
