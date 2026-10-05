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
- **Dynamic arrays for all creative variety.** All creative parameters (tones, examples, phrases, styles) live in static arrays on the LoveEngine class, sampled via `_pickRandom()`/`_pickWeighted()`, and extended by the LLM every 5th post via `_maybeExtendLists()`. Never hardcode creative content directly into prompt strings.
- **The closing beat is a pool, not a clause.** The third beat of the post prompt used to be
  hardcoded, first as "…and want to send it to someone they care about" and then as "…and leave
  something behind that stays." Both were replaced after the model latched on and every post closed
  the same way — swapping one fixed phrase for another fixed phrase just moves the monotony. The beat
  now comes from `POST_BEATS`, grown by `_maybeExtendLists()` and rotated by `_pickWeighted`
  against `recentBeats`. See the closing-beat note under Robustness.
- **Image grounding (`_deriveVisualBrief`).** The image prompt used to accept a `postText` argument
  and ignore it: the scene was built from plan and seed only, so the post's own imagery never
  reached it. A post reading *"A glass of water sits on the edge of a table"* rendered a crystalline
  nebula with no glass, water or table. One extra LLM call now reads the story and returns 3–5
  photographable anchors plus material and light, and only those nouns reach CLIP. Measured
  working: brief `plane, gold, metal` produced an aircraft wing; `shore, tide, water` produced a
  tidal pool. **It does not achieve literal correspondence.** The anchors compete with ~35 words of
  fixed technique/palette/composition boilerplate — the aesthetic the account is built on — so a
  candle post produced a beautiful image with no candle in it. Treat "match the post" as mood
  correspondence, which is what currently works, not depiction.
- **Six growable creative pools**, all unbounded and self-widening: `edgeVocabulary`,
  `planVibes`, `directorVibes`, `compositionSlots`, `openingForms`, `postBeats`. Caps are set to
  50000, which turns them from "evict oldest when full" into "never evict" — hand-written seeds
  persist and the pools only widen. Real growth is bounded by head-word diversity, not the cap:
  a head may appear `HEAD_MAX_PER` (2) times, so a pool saturates when the model runs out of new
  heads. Growth adds at most 6 per 5th transmission and never grows a prompt — pools are sampled
  one or eight at a time and the generation prompt only ever shows the last few entries.
- **Two distinct prompt modes:** `SOCIAL_POST_PROMPT` for text posts/replies/DMs, `VIDEO_VOICEOVER_PROMPT` for video voiceovers. Both share the same tonal rotation system.

### Content Generation
- Posts use deterministic tone rotation from `LoveEngine.TONES` array (cycles by transmission number)
- Subliminal phrases use `PHRASE_TERRITORIES` for emotional variety and `PHRASE_STRUCTURES` for structural variety
- The `_maybeExtendLists()` system grows all arrays over time via LLM generation + localStorage persistence. It is real as of 2026-10-03 and the closing-beat pool is its first user; the other documented lists still have static contents.

### Video Pipeline
- 5-scene production: scenes + voiceover + music generated as one unified brief
- Splice uses MessageChannel frame pump (immune to background tab throttling)
- Per-scene stall detection (3s) and wall-clock timeout (15s) force-advance stuck scenes
- 30s video-time trim with wall-clock fallback for invalid video.duration
- 53 GLSL shaders x 20 animations = 1,060 unique caption combinations

### Deploy
- `bash deploy.sh` — auto-increments the build number and runs `npx firebase-tools deploy --only hosting`.
  The script itself was deleted in `d9bd66d8` ("deleted most of L.O.V.E 1") while this section still
  documented it; restored on 2026-10-03, hardened. It takes the higher of `public/version.json` and
  the `build-version` stamp in `public/index.html`, because those two had drifted (json said 91 while
  the page read 94) and the original trusted the json alone — it would have bumped to 92 and stamped
  the page *backwards*.
- `firebase-tools` is not installed globally; the script pulls it through `npx`. **Deploying needs
  `firebase login` first** — there are no stored credentials in `~/.config/configstore`, so the
  deploy halts at an interactive browser OAuth step that cannot be automated.
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
  **Welcome posts are OFF by default; `--welcome` opts back in.** Follow-back always runs, so the flag
  only gates the welcome. While off, people are followed back but not queued, so turning it back on
  applies only to later arrivals and no backlog dumps itself on resume; anything already queued is
  held and reported rather than dropped. Off by default because the welcome render is the expensive
  half of the interaction and the prompt asks for the phrase "rendered in the scene", which SDXL
  renders as legible-looking nonsense — the first real welcome came back as `YS / YOU / A PME` in
  place of the signal `YOU ARE HOME`. `--no-welcome` is still accepted as an explicit spelling of
  the default, so older scripted invocations keep working.
- **GPU sharing**: the LLM (Ollama) and SDXL cannot share the 6GB VRAM; `love-ai.sh image` and `love-cli.mjs` unload the Ollama model first. The SRBMiner miner (`~/epic-mining/start_epic_ubuntu.sh`) also holds ~2GB VRAM and auto-respawns — stop the wrapper script, not just the miner.
- **Local-mode gaps**: video, TTS, and music throw — only text + image posts are supported.
- **Subliminal text in images**: NOT rendered locally. The webapp prompted gpt-image (cloud)
  to weave the phrase into the scene; local diffusion models cannot render readable text.
  A PIL compositing fallback exists (`~/ai/overlay_text.py`) but is disabled by design.

## Robustness notes
- **qwen3 responds to structure, not instructions.** This is the single most useful thing learned
  about this model, and every instance of it was found by measuring rather than reasoning:
  - **Asking for variety does nothing. Prescribing it works.** Told to "give each a different
    register", beat generation returned 7 lines opening `let...` out of 20. Assigning a category per
    line took distinct verbs from 1-in-4 to 4-in-4.
  - **Naming a word to avoid primes it.** Telling the generator `let` was over-used took it from 7 to
    13. Quoting the offending phrase in the fragment cap's rejection feedback did the same: while it
    named "you're already", that fragment climbed 25% → 40% of the ring. Feeding back a live tally
    reads as a suggestion for the same reason (7 → 10).
  - **Showing the existing list gets it echoed back.** The opening-form generator returned all six
    seeds unedited. Shortening the visible list to four unblocked growth.
  - **Hard caps beat instructions.** The beat verb cap (reject, don't ask) is the only mechanism here
    that has reliably held a distribution.
- **Log every mechanism that can silently do nothing.** The single most expensive recurring bug this
  session was an invisible failure: the fragment cap reported nothing when it rejected, and nobody
  could tell from the outside whether it was working. The pools now log `unchanged — N candidate(s),
  none accepted` and `beat pool unchanged — every candidate used a saturated verb`. A silent no-op is
  indistinguishable from a feature that does not exist.
- **Distrust counters that report the wrong thing.** Three separate times a log line counted the
  wrong quantity and I read it as truth: `beat pool 20 → 20 (+6)` when the cap had truncated every
  push away, `+5 accepted` for a pool that did not grow, and `_tooSimilar(x, v, 0.7)` where the third
  argument did not exist and the threshold was hardcoded at 0.5. When a number is the evidence,
  check what it actually counts.
- **Bounded growth beats no growth; unbounded beats both.** The beat pool's starvation escape
  accepted over-represented beats when the verb cap rejected everything, and because that happens
  most cycles the leak was cumulative — `your` went 3 → 5, then 3 → 6. Removed. A pool that stops
  growing is a lesser failure than one drifting toward a single verb.
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
- **The `Xs elapsed` in the `ready` line is per-post**, measured from a `t0` declared *inside* the
  post loop. It used to be declared outside, so the number grew by one post-time per post and read
  like a runaway: a steady ~290s/post logged as 275.6 → 567.5 → 855.0 → 1157.4s. The end-of-run
  summary keeps its own `runStart` for the cumulative total. If the per-post figure ever looks like
  growth again, confirm against the `output/transmission-N.png` mtime intervals before believing it.
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
- **Alt text is the image prompt, capped at 3000** (Bluesky's limit). `postOne()` takes it and
  passes it to `createPost()`; without it, `createPost` falls back to `text.slice(0, 100)` and every
  image ships with the post body as its description. `generateWelcome` clamps its own prompt at
  4000, so the trim has to happen here or an oversized welcome fails the upload after the render.
- **Closing-beat pool** (`_maybeExtendLists()` → `_pickBeat()`): every 5th transmission the LLM adds
  up to 6 beats, rejecting near-duplicates by trigram overlap and by over-used main verb, capped at
  24, persisted as `love_beat_pool`. **Additive and non-destructive by construction** — a failed,
  empty, or all-duplicate response leaves the pool untouched, so generation can never make a post
  worse than not having tried. The cap evicts oldest-first, so once the pool is full the curated
  seeds are replaced by generated beats over time. Two things to know before trusting it:
  - **Verb diversity is enforced in code** (`BEAT_VERB_CAP = 3`): a beat whose main verb already
    opens three pool entries is rejected. Prompt-level control was tried first and is not
    sufficient — naming "let" as the verb to avoid took it from 7 to 13, because telling a model
    which word to move away from primes it, and feeding the live verb tally back did the same more
    weakly (7 → 10) since a high count reads as a suggestion. Assigning a form per line helped
    (1 → 3-4 distinct verbs per 4 beats) but did not close it. If every candidate is rejected, the
    verb cap relaxes for one pass so the pool degrades to "samey" rather than freezing; dedupe
    still applies.
  - **The CTA guard is prompt-only**, by deliberate choice. There is no code filter dropping
    forwarding verbs. Measured across every pool built so far: zero beats have contained one. If
    that changes, add the filter — generation drifts, filtering doesn't.
- **`_pickWeighted` returns a single item, not an array.** It is easy to write `[0]` on the result
  and silently get the first *character* — which for any string starting `"..."` is always `.`.
- `_loadVarietyMemory` parses a missing key as `"[]"` and assigns it, so any list that must ship with
  seed content (the beat pool does) gets its seeds discarded on a fresh install and has to be
  normalised back after load.
- **Repetition has three layers, and they are not interchangeable.** `_wordTrigrams` takes raw
  3-word windows (it does *not* strip stopwords — an earlier note here said it did; that was wrong).
  A repeated sentence does produce trigrams, so the original critic's miss was never about token
  density: its aggregate test is `reused / size(newTrigrams) > 0.4`, a **whole-post ratio**. One
  repeated sentence is ~2 trigrams out of ~26 — 8%, far under the threshold. A two-word phrase like
  "just breathe" produces no trigram at all. Hence:
  - `_repeatsSentenceFromRing` — direct ≥3-word sentence comparison, catches the diluted case.
  - `_fragmentOverused` — a **frequency cap** (`FRAGMENT_FREQ_CAP = 0.20`): if a distinctive 2-word
    fragment is already in ≥20% of the ring, a post using it is rejected. Only fragments containing
    a content word count (`_isDistinctiveFragment`); "in your", "like a", "you are" are excluded, or
    the cap would reject most posts and starve generation. Cap value was measured, not guessed.
  - `_repetitionCost` — soft ranking, so the loop prefers novel phrasing and can always fall back.
- **The content loop ranks candidates instead of returning the last attempt.** The trigram guard and
  the boredom critic are both skipped on the final attempt (`attempt < MAX_RETRIES - 1`), so when the
  critic rejects three attempts for cliché the fourth is accepted unchecked — and "you're already
  glowing" is itself the cliché being chased. Ranking is what closes that.
- **Measured effect of the repetition guards (2026-10-03).** A full 20-post ring was snapshotted
  pre-fix, then the ring was allowed to refill completely and compared like for like:

  | metric | pre-fix | post-fix |
  |---|---|---|
  | `just breathe` | 5/20 (25%) | **0/20** |
  | ≥3-word sentence repeats (leave-one-out) | 13/20 (65%) | **6/20 (30%)** |
  | ≥2-word sentence repeats | 15/20 (75%) | 9/20 (45%) |
  | any use of "breathe" | 8/20 (40%) | 1/20 (5%) |
  | most-repeated 2-word fragment | `just breathe` ×5 (25%) | `you're here` ×4 (20%) |

  The ≥3-word guard does real work: verbatim sentence reuse more than halved and the target phrase
  is gone. **The fragment ranking does not bound repetition, it redistributes it.** The leading
  short fragment moved from `just breathe` to `you're here` at a comparable rate, and a new cliché
  appeared (`you are already enough`, 0/20 → 2/20). Half weight is too weak to stop migration. If
  this is to be fixed rather than observed, the beat pool already contains the pattern that works —
  a hard cap on how many entries may share a property — applied as a fragment-frequency cap over
  the ring. The risk is starvation: `a breath` opens 30–40% of posts, so a cap that rejects it
  outright would leave the engine unable to write a post. Measure against a full ring before
  trusting any such cap. One full refill is a single sample at one temperature; it shows the
  mechanism moves the metric, not that the new equilibrium holds over days.
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
