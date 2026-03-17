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
