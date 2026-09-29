# Local AI Stack for L.O.V.E

Runs entirely on this machine (GTX 1060 6GB). Replaces the Pollinations API.

The scripts in this directory are versioned in the repo. The weights and the
`imgenv` venv they load are not — those live in `~/ai` (override with
`LOVE_AI_HOME`).

## Stack
- **LLM**: Ollama 0.34.4 at `~/ai/ollama`, model `qwen3:8b` (5.2GB, chosen by side-by-side
  test — see `~/ai/compare/text_models.md`). OpenAI-compatible endpoint at
  `http://127.0.0.1:11434/v1/chat/completions` (drop-in for `generateText`; no thinking-mode
  pollution via /v1). Also pulled for comparison: qwen2.5:7b, gemma3:4b, llama3.1:8b,
  gemma4:e2b (delete unused ones from `ollama rm <model>` to reclaim ~15GB).
- **Images (rotation)**: SDXL base (`~/ai/sdxl`), LEOSAM HelloWorld v7 (`~/ai/leosam`),
  RealVisXL V5 (`~/ai/realvis`) — one picked randomly per image in `generate_image.py`
  and `render_batch.py` (Euler A scheduler, CFG 7, fp16).
  Also installed but unused: BigASP v2 (`~/ai/bigasp`, pony-arch).
  Model comparison sheets: `~/ai/compare/` (variety_sheet.jpg, contact_sheet.jpg).
- **Subliminal text in images**: not rendered — local diffusion models can't do readable
  text. A PIL compositing fallback exists (`~/ai/overlay_text.py`) but is disabled.
- **Miner conflict**: SRBMiner (epic-mining) must stay off during generation; it
  auto-respawns via `~/epic-mining/start_epic_ubuntu.sh` — kill that wrapper first.
- Ollama and SDXL cannot share 6GB VRAM: `love-ai.sh image` and `love-cli.mjs`
  unload the LLM model before rendering.

## Usage
Run from the repo root; `love-ai.sh` resolves the scripts next to itself.
```bash
ai/love-ai.sh text  "system prompt here" "user prompt here"   # -> text on stdout
ai/love-ai.sh image --prompt "..." --out out.png [--w N --h N --steps N --seed N --negative "..."]
~/ai/imgenv/bin/python ai/render_batch.py jobs.json           # batch render, random model per job
node love-cli.mjs [--post] [--batch N]                       # full pipeline
~/ai/imgenv/bin/python ai/memdiag.py [jobs.json]             # VRAM usage per render stage
```

## Diagnosing OOM
The card is tight enough that where memory sits decides whether a render
survives: the UNet alone needs ~3.7GB. `memdiag.py` prints allocation after
each stage, so a leak shows up as a number instead of a stack trace. If the
text encoders show up as resident after encoding, something ran them through
accelerate's offload hook instead of the CPU path — see the notes in
`long_prompt.py` and `../CLAUDE.md`.

## Performance measured
- LLM post text: ~8-17s (16 tok/s, split CPU/GPU on the 6GB card)
- Image: ~54s at 768px/25 steps warm; ~3.5 min at 1024px/28 steps
- Batch of 10 posts (10 texts + warm image renders + posts): ~50 min end-to-end

## Reinstall notes
Everything lives in `~/ai` (~38GB: three image models + bigasp, ollama + 5 models, venv).
Deleted after setup: HuggingFace cache, raw pony .bin, ollama tarball. To re-download:
- Ollama: https://github.com/ollama/ollama/releases (Linux amd64 tarball, no sudo needed)
- SDXL base: `stabilityai/stable-diffusion-xl-base-1.0` (fp16 variant)
- LEOSAM v7: `misri/leosamsHelloworldXL_helloworldXL70` (unet fp16)
- RealVisXL V5: `SG161222/RealVisXL_V5.0` (unet fp16)
- BigASP v2: `John6666/big-asp-v2-sdxl`
