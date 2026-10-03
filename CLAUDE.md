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
- **Ollama runs as a systemd *user* service** (`~/.config/systemd/user/ollama.service`, `systemctl
  --user`), not a system service and not a bare background process. `Linger=yes` is already set on
  this account, so it starts at boot with no login. It sets `OLLAMA_MODELS=/home/raver1975/ai/ollama-models`
  and that line is load-bearing: Ollama otherwise defaults to `~/.ollama/models`, which is **empty**
  on this box, so it comes up healthy on `/api/version` and then answers every request with
  `model 'qwen3:8b' not found` (the serve log says `total blobs: 0`). If the model store ever moves,
  the unit is the thing to update.
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
  - **The `Token indices sequence length is longer ... (90 > 77)` warning is expected, not a
    failure.** transformers raises it at *tokenization* time, not inference: it fires whenever
    `len(ids) > model_max_length` and no `max_length` was passed, and `_chunk_ids`/`num_chunks` call
    the tokenizer precisely to count tokens without truncating — so every prompt over 77 tokens
    trips it. The 90-id sequence is then split into 75-token chunks and never reaches an encoder, so
    the indexing errors it predicts cannot occur. Confirm from the render line: `tokens=90
    embeds=(1, 154, 2048)` is 2 chunks x 77, exactly as designed. It prints twice per prompt because
    SDXL has two tokenizers and the warning is once per tokenizer instance.
  - The negative needs its own pooled vector from encoder 2 (SDXL's unconditional branch); reusing
    the positive one conditions the negative branch on the prompt it is meant to steer away from.
  - Both text encoders return different types: `text_encoder` is a plain `CLIPTextModel`
    (`pooler_output`), `text_encoder_2` has the projection (`text_embeds`). Only encoder 2's
    pooled vector is used for SDXL.
- **Entry point**: `ai/love-ai.sh text|image ...` — see `ai/README.md`
- **CLI app**: `node love-cli.mjs [--post] [--skip-image] [--once]` — runs the full
  LoveEngine pipeline locally (state in `.love-state.json`, credentials in gitignored `.env`).
  **Strictly sequential: one post at a time — text → image → published → next, forever.** The
  loop runs until interrupted; `--once` stops after a single post. There is **no pause between
  successful posts** — the ~6 minute SDXL render is the natural spacing. This replaced a
  three-phase design that queued N texts, rendered them in one warm SDXL session, then posted in
  a burst; that one stalled ~50 min before anything reached Bluesky and lost every render in the
  batch when a process died mid-render. The cost of going sequential is a full SDXL model load per
  image plus an Ollama unload/reload between each post's text and image, but it is cheaper than
  expected: 3477s for 10 posts (~348s each) against ~50 min for the warm-session batch, roughly
  16% slower rather than the 2.4x an earlier version of this note claimed. A render is almost
  entirely diffusion steps, so the ~35s model load is a small fraction of it. In exchange, posts go
  live immediately and a failure costs one post instead of ten. A failed post is caught and the
  loop continues.
  **Failure backoff** (`loop.mjs`): no pause on success, but consecutive failures wait
  60s → 120s → 240s … capped at 15 minutes, and any success resets the streak. Without it, two
  fast failures in a row (bad credentials, a throw before the first network call) would retry
  immediately in a tight loop. Note a failing *post* still costs ~35s before the backoff starts,
  because `generateText` retries 3x internally (5s, 10s) inside the outer loop's 2 attempts.
  Re-encodes oversized PNGs to JPEG (Bluesky 2MB blob cap).
  `ai/render_batch.py` is no longer spawned by the CLI; it survives as the manual recovery tool for
  re-rendering a `jobs.json` written before this change.
  Scheduled runs: `./love-run.sh [extra flags]` — takes the single-instance lock, tees output to
  `love-run.log`, and runs `love-cli.mjs --post` forever. A second launch refuses to start.
- **Follow-back + welcome** (`doFollowBack()` in `love-cli.mjs`): one scan per post cycle, run at the
  END of the iteration so a bad scan can never cost a post. It follows back anyone not followed, and
  sends a welcome post (`generateWelcome()` = 1 LLM call + a full SDXL render). Budget is **one
  welcome per scan**; the rest queue in `love_pending_welcomes`. The queue is load-bearing — following
  someone removes them from `getUnfollowedFollowers()`, so a deferred welcome would otherwise never be
  seen again. The first scan only follows back and records without posting (mirroring the webapp's
  `isFirstFollowScan`), so shipping this doesn't spam welcomes at the existing backlog. Welcomed/
  followed records reuse `engine.interactions`, which works headless because the CLI shims
  `localStorage` to `.love-state.json`. Everything logs under `[follow]`. The webapp's
  `doFollowBack()` only ran while the dashboard tab was open, so a headless CLI run followed nobody.
  `--no-welcome` pauses welcome posts without pausing follow-back: people are still followed, but not
  queued, so re-enabling applies only to later arrivals and no backlog dumps itself on resume.
- **GPU sharing**: the LLM (Ollama) and SDXL cannot share the 6GB VRAM; `love-ai.sh image` and `love-cli.mjs` unload the Ollama model first. The SRBMiner miner (`~/epic-mining/start_epic_ubuntu.sh`) also holds ~2GB VRAM and auto-respawns — stop the wrapper script, not just the miner.
- **Local-mode gaps**: video, TTS, and music throw — only text + image posts are supported.
- **Subliminal text in images**: NOT rendered locally. The webapp prompted gpt-image (cloud)
  to weave the phrase into the scene; local diffusion models cannot render readable text.
  A PIL compositing fallback exists (`~/ai/overlay_text.py`) but is disabled by design.

## Robustness notes
- **A CUDA driver mismatch silently CPU-falls-back Ollama and looks like an LLM bug.** On 2026-09-29
  an `apt` run cycled `nvidia-driver-570` → purge → reinstall → purge → `install nvidia-driver-535`,
  and every attempt re-pulled the 580 packages as automatic dependencies. The result was a kernel
  module of 535.309.01 (loaded at the last boot) against 580.178.04 userspace libraries, with no
  `libcuda.so.535*` on disk at all. Nothing errored loudly: `nvidia-smi` reported an NVML mismatch,
  PyTorch raised `Error 804` with `torch.cuda.is_available() == False`, and **Ollama just fell back to
  the CPU** — a 2-token reply took 83.9s (0.54 tok/s prompt eval) instead of ~1s. Every LLM call then
  blew the 120s `AbortSignal.timeout` in `love-cli.mjs:83`, and posts 25–36 were lost while the backoff
  climbed to 900s. The tell is `llama-server` pegged at ~860% CPU with the model resident in RAM.
  **Fix:** reboot — `dkms` had already built the 580 module, so booting it matched the userspace with
  no package surgery. Afterwards the 535 branch was purged and the 580 stack marked
  `apt-mark manual`; that marking is what stops the next `apt upgrade` from reinstalling 535 userspace
  over the 580 ones and repeating the whole thing. Note that purging 535 makes apt consider the *entire*
  580 stack orphaned (it entered as an automatic dependency of 535) and offer to delete
  `nvidia-dkms-580` — never run `apt autoremove` before the `apt-mark manual`.
- **The `Xs elapsed` in the `ready` line is cumulative, not per-post.** `t0` is declared at
  `love-cli.mjs:280`, *outside* the post loop, so the number grows by one post-time per post and reads
  like a per-post timer. A steady ~290s/post shows up as 275.6 → 567.5 → 855.0 → 1157.4s. The real
  per-post cost is flat, confirmed three ways: the cumulative deltas (275.6/291.9/287.5/302.4), the
  `output/transmission-N.png` mtime intervals (+292/+287/+302), and a phase trace (text ~94s + render
  ~181s). Don't diagnose a slowdown from that field without differencing it first.
- **`love-run.log` is shared by every run and full of `\r`.** tqdm writes carriage returns to stderr
  alongside node's `console.error`, so `grep -E '^\[seq\]'` silently misses lines that got glued to a
  progress bar, and `awk` ranges match *earlier* runs' identically-numbered posts. Scope to the run
  you care about first: `tr '\r' '\n' < love-run.log | awk '/starting continuous mode/{n=NR} {l[NR]=$0} END{for(i=n;i<=NR;i++) print l[i]}'`.
  For per-post durations prefer the PNG mtimes, which are immune to all of this.
- **`app.bsky.graph.*` endpoints cap at 100 per page and return no total — always paginate.**
  `getUnfollowedFollowers()` originally diffed `getFollowers` against `getFollows` with `limit=100`
  on each and was silently wrong once this account passed 100 on either side (207 followers, 244
  following). Comparing the first 100 of each list is a diff of two unrelated windows: it reported
  **12 people we already followed** as unfollowed and **zero** real ones. Because the caller follows
  the returned list and queues welcome posts for it, that would have spammed welcomes at people who
  followed months ago. Fixed with `_fetchAllPages()`, which cursor-walks both endpoints to exhaustion
  under a runaway `cap` (there is no total to bound against). Any future graph query here needs the
  same treatment — and remember that a single-page read of this account looks like 87/98 when the
  truth is 207/244.
- `love-run.sh` does **not** reap its SDXL child. Killing the run mid-render orphans a
  `generate_image.py` that keeps holding ~5.8GB of VRAM at 100% util, so the next run fails to
  allocate. Check `pgrep -f generate_image.py` after stopping a run (mind that the pattern matches
  your own shell — use the PID from `ps` rather than `pkill -f`).
- qwen3 occasionally emits off-schema JSON at high LFO temperatures: the creative seed
  falls back to default fields, and generation retries a post once before skipping
  (a single bad generation never kills a run).
- Every LLM call is capped (`num_predict`) and bounded by an `AbortSignal.timeout`. Without
  the timeout a stalled request never rejects, so the retry loop only ever sees errors and a
  hang waits forever — this is what wedged a continuous run in 2026-09-27.
- `love-run.sh` holds an exclusive `flock` on `.love-run.lock` and refuses to start a second run.
  Two concurrent runs collide on the 6GB card: the loser gets `failed to allocate Vulkan0 buffer`
  and an Ollama 500 (2026-09-29). Three details are load-bearing — the fd is opened `>>` so a
  refused launch does not truncate the pid it is about to report; every child closes it with `9>&-`
  so an inherited fd cannot keep the lock alive in an orphaned `node`; and the file is never
  unlinked, because `flock` locks the inode rather than the path, so removing it would let two
  launchers each hold "the lock" on different inodes.
- A render can also die with *no* traceback at all: no `BATCH_DONE`, no `=== stopped (exit N) ===`
  line from `love-run.sh`, and no kernel OOM record. On 2026-09-28 a batch stopped at step 11/28 exactly
  that way — `love-run.sh` runs `node` under `tee` in a plain shell with no `nohup`/`setsid`, so
  closing the terminal takes the whole tree down. Start long runs detached (`setsid nohup ./love-run.sh &`).
  Distinguish the two causes by checking `/var/log/kern.log` for `Out of memory: Killed process` first —
  this box OOM-killed a python process at 18:34 (29GB RSS) and another at 19:28 (15GB RSS) that same
  day, so a genuine host-RAM kill is a real possibility here, not a hypothetical.

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
