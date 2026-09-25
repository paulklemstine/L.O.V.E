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
- **LLM**: Ollama + `qwen2.5:7b-instruct-q4_K_M`, OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1/chat/completions` (drop-in for `generateText`)
- **Images**: Pony Diffusion V6 XL (fp16) in `~/ai/pony` (SDXL-arch, reuses base SDXL
  encoders/VAE), 6GB-VRAM tuned (drop-in for `generateImage`). Prompts get Pony's
  score-tag convention (`score_9, score_8_up...` + score negatives, CFG 7) added in
  `~/ai/generate_image.py` and `~/ai/render_batch.py`. Base SDXL kept at `~/ai/sdxl`.
- **Entry point**: `~/ai/love-ai.sh text|image ...` — see `~/ai/README.md`
- **CLI app**: `node love-cli.mjs [--post] [--skip-image] [--batch N]` — runs the full
  LoveEngine pipeline locally (state in `.love-state.json`, credentials in gitignored `.env`).
  `--batch N` queues N texts, renders all images in one warm SDXL session (~2.4x faster),
  then posts in a burst. Re-encodes oversized PNGs to JPEG (Bluesky 2MB blob cap).
  Scheduled runs: `./love-run.sh [batches] [batch_size]` (logs to `love-run.log`).
- **GPU sharing**: the LLM (Ollama) and SDXL cannot share the 6GB VRAM; `love-ai.sh image` and `love-cli.mjs` unload the Ollama model first. The SRBMiner miner (`~/epic-mining/start_epic_ubuntu.sh`) also holds ~2GB VRAM and auto-respawns — stop the wrapper script, not just the miner.
- **Local-mode gaps**: video, TTS, and music throw — only text + image posts are supported.
