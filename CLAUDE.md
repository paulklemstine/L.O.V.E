# L.O.V.E. — Development Guide

## Project Structure
```
public/                  # Firebase-hosted web app
  js/love-engine.js      # Core engine: content generation, video pipeline, social interactions
  js/trippy-text.js      # WebGL caption renderer: 53 GLSL shaders + 20 animations
  js/pollinations.js     # API client: text (GPT-5 Mini), image (FLUX), video (grok), TTS, music
  js/bluesky.js          # AT Protocol client: posts, video upload, replies, DMs, follows
  js/app.js              # Dashboard controller: UI, activity log, post history
  index.html             # Control panel UI
  test-video.html        # Video splice test rig
deploy.sh                # Auto-incrementing Firebase deploy
firebase.json            # Firebase hosting config
```

## Key Rules

### Prompt Engineering
- **Positive instructions only.** Frame everything as what TO do. Never use "do not", "never", "avoid", "banned" in LLM prompts.
- **Dynamic arrays for all creative variety.** All creative parameters (tones, examples, phrases, styles) live in static arrays on the LoveEngine class, sampled via `_pickRandom()`, and extended by the LLM every 5th post via `_maybeExtendLists()`. Never hardcode creative content directly into prompt strings.
- **Two distinct prompt modes:** `SOCIAL_POST_PROMPT` for text posts/replies/DMs, `VIDEO_VOICEOVER_PROMPT` for video voiceovers. Both share the same tonal rotation system.

### Content Generation
- Posts use deterministic tone rotation from `LoveEngine.TONES` array (cycles by transmission number)
- Subliminal phrases use `PHRASE_TERRITORIES` for emotional variety and `PHRASE_STRUCTURES` for structural variety
- The `_maybeExtendLists()` system grows all arrays over time via LLM generation + localStorage persistence

### Video Pipeline
- 5-scene production: scenes + voiceover + music generated as one unified brief
- Splice uses MessageChannel frame pump (immune to background tab throttling)
- Per-scene stall detection (3s) and wall-clock timeout (15s) force-advance stuck scenes
- 30s video-time trim with wall-clock fallback for invalid video.duration
- 53 GLSL shaders x 20 animations = 1,060 unique caption combinations

### Deploy
- `bash deploy.sh` — auto-increments build number, deploys to Firebase
- Always commit + push before deploying
- Live at https://l-o-v-e.web.app

## Common Commands
```bash
git add <files> && git commit -m "message" && git push && bash deploy.sh
```

## Local AI Stack (replaces Pollinations API)
- **LLM**: Ollama + `qwen3:8b` via the native endpoint `http://127.0.0.1:11434/api/chat`. `generateText` must pass `think: false` — qwen3 otherwise emits a long <think> block that the OpenAI-compatible `/v1/chat/completions` endpoint returns in a separate `reasoning` field, leaving `content` empty after burning the entire token budget. Neither `chat_template_kwargs.enable_thinking` nor a `/no_think` suffix suppresses it on `/v1`; only the native endpoint's `think: false` works (~12s → ~1s per call). Previous: `qwen2.5:7b-instruct-q4_K_M`
- **Images**: 3-model rotation — SDXL base (`~/ai/sdxl`), LEOSAM HelloWorld v7 (`~/ai/leosam`),
  RealVisXL V5 (`~/ai/realvis`) — one picked randomly per image in `ai/generate_image.py`
  and `ai/render_batch.py` (Euler A scheduler, CFG 7). Model choice is compared/tested via
  `~/ai/compare/` (contact sheets). Unused-but-installed: BigASP v2 (`~/ai/bigasp`).
- **Long prompts (CLIP 77-token window)**: generated image prompts run 90–116 CLIP tokens, so
  diffusers silently truncated ~25% of every one — the tail, which is exactly where
  `_generateImagePrompt` puts the palette and composition slot (3 of 13 composition markers were
  being dropped). `ai/long_prompt.py` splits the prompt into 75-token chunks, encodes each, and
  concatenates the `last_hidden_state` sequences, so a 115-token prompt yields 154 cross-attention
  tokens instead of 77. Wired into both `generate_image.py` and `render_batch.py`.
  - 77 is a learned position-embedding table (`max_position_embeddings: 77` in both text encoder
    configs), not a setting — raising it in config without retraining just breaks the shape.
  - The negative prompt is padded with EOS-filled chunks to match the positive side's chunk count;
    classifier-free guidance requires both embeddings to share a sequence length.
  - **Every prompt encodes on CPU**, short ones included, bypassing accelerate's offload hook via
    the module's `_old_forward`. The encoders left resident by the hook cost ~1.9GB (encoder 2
    alone is 1325MB) and the UNet needs 3744MB, so together they overrun a 6GB card. Short-circuiting
    single-window prompts to `pipe.encode_prompt` for speed looks harmless but OOMs: under
    `model_cpu_offload` the hook only returns components to CPU in `maybe_free_model_hooks()` at the
    end of a full `pipe(...)` call, and a standalone `encode_prompt` never triggers it. That is how
    a 6-token prompt died at 1890MB resident while a 107-token one rendered fine at 1MB — the long
    production prompts hid it. The CPU path costs a couple of seconds against a multi-minute render.
  - The negative needs its own pooled vector from encoder 2 (SDXL's unconditional branch); reusing
    the positive one conditions the negative branch on the prompt it is meant to steer away from.
  - Both text encoders return different types: `text_encoder` is a plain `CLIPTextModel`
    (`pooler_output`), `text_encoder_2` has the projection (`text_embeds`). Only encoder 2's
    pooled vector is used for SDXL.
- **Entry point**: `ai/love-ai.sh text|image ...` — see `ai/README.md`
- **CLI app**: `node love-cli.mjs [--post] [--skip-image] [--batch N]` — runs the full
  LoveEngine pipeline locally (state in `.love-state.json`, credentials in gitignored `.env`).
  `--batch N` queues N texts, renders all images in one warm SDXL session (~2.4x faster),
  then posts in a burst. Re-encodes oversized PNGs to JPEG (Bluesky 2MB blob cap).
  Scheduled runs: `./love-run.sh [batches] [batch_size]` (logs to `love-run.log`).
- **GPU sharing**: the LLM (Ollama) and SDXL cannot share the 6GB VRAM; `love-ai.sh image` and `love-cli.mjs` unload the Ollama model first. The SRBMiner miner (`~/epic-mining/start_epic_ubuntu.sh`) also holds ~2GB VRAM and auto-respawns — stop the wrapper script, not just the miner.
- **Local-mode gaps**: video, TTS, and music throw — only text + image posts are supported.
- **Subliminal text in images**: NOT rendered locally. The webapp prompted gpt-image (cloud)
  to weave the phrase into the scene; local diffusion models cannot render readable text.
  A PIL compositing fallback exists (`~/ai/overlay_text.py`) but is disabled by design.

## Robustness notes
- qwen3 occasionally emits off-schema JSON at high LFO temperatures: the creative seed
  falls back to default fields, and batch generation retries a post once before skipping
  (a single bad generation never kills a batch).
- Every LLM call is capped (`num_predict`) and bounded by an `AbortSignal.timeout`. Without
  the timeout a stalled request never rejects, so the retry loop only ever sees errors and a
  hang waits forever — this is what wedged a continuous run in 2026-09-27.

## Repo layout: local image scripts
- `ai/generate_image.py`, `ai/render_batch.py`, `ai/long_prompt.py`, `ai/love-ai.sh`,
  `ai/memdiag.py` and `ai/README.md` are versioned in this repo.
  Model weights (~25GB across `sdxl`/`leosam`/`realvis`), the `imgenv` venv and the Ollama
  install are NOT — they live in `~/ai` and are located via `LOVE_AI_HOME` (defaults to
  `~/ai`). `love-cli.mjs` invokes the repo copies using that venv, and `love-ai.sh` resolves
  the scripts beside itself, so neither hardcodes a checkout path. Earlier copies of the
  Python scripts and of `love-ai.sh` sat in `~/ai` next to the weights and are now deleted —
  an unversioned second copy is how the CLIP truncation fix nearly got bypassed.
- `ai/memdiag.py` prints VRAM per render stage. Run it first when a render OOMs on the 6GB
  card: resident text encoders mean something took the hooked GPU path instead of the CPU one.
