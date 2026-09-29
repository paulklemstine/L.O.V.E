#!/usr/bin/env python3
"""Report VRAM at each stage of a single SDXL render.

The 6GB card is tight enough that where memory sits decides whether a render
survives: the UNet alone needs ~3.7GB, so anything left resident by the text
encoders (~1.9GB, encoder 2 being 1325MB of that) is the difference between
working and CUDA OOM. This prints allocation after each stage so a regression
shows up as a number rather than a stack trace.

Usage:
  memdiag.py [jobs.json]     # defaults to a built-in prompt pair
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_prompt import encode_long_prompt, num_chunks

AI_HOME = os.environ.get("LOVE_AI_HOME", os.path.join(os.path.expanduser("~"), "ai"))
MODEL = os.path.join(AI_HOME, "sdxl")  # fixed, so numbers are comparable run to run
OUT = "/tmp/memdiag.png"

DEFAULT_JOBS = [{
    "prompt": ("a quiet coastal road at dusk, wet asphalt reflecting amber streetlight, "
               "cinematic, shallow depth of field, muted teal and warm amber palette, "
               "anamorphic, 35mm, a lone figure in a heavy coat walking away from camera, "
               "low horizon line, rolling fog off the water, deep shadows, high contrast, "
               "shot on Kodak Portra, gentle film grain, soft natural light"),
    "negative": "blurry, low quality, deformed, watermark, text artifacts",
}]


def main():
    jobs = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else DEFAULT_JOBS
    job = jobs[0]
    prompt, negative = job["prompt"], job.get("negative")

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    def report(tag):
        print(f"{tag:22s} {torch.cuda.memory_allocated()/2**20:7.0f}MB alloc / "
              f"{torch.cuda.memory_reserved()/2**20:7.0f}MB reserved", flush=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    report("after load+offload")

    ntok = len(pipe.tokenizer(prompt, truncation=False,
                              add_special_tokens=False)["input_ids"])
    print(f"{'':22s} {ntok} tokens -> {num_chunks(pipe.tokenizer, prompt)} chunk(s)", flush=True)

    pe, npe, pp, npp = encode_long_prompt(prompt, negative, pipe, pipe._execution_device)
    report("after encode")
    for name in ("text_encoder", "text_encoder_2", "unet", "vae"):
        mod = getattr(pipe, name)
        held = sum(p.numel() * p.element_size() for p in mod.parameters()
                   if p.device.type == "cuda")
        if held:
            print(f"{'':22s} {name} resident on GPU: {held/2**20:.0f}MB", flush=True)
    torch.cuda.empty_cache()
    report("after empty_cache")

    try:
        img = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=pp,
                   negative_pooled_prompt_embeds=npp, width=1024, height=1024,
                   num_inference_steps=4, guidance_scale=7.0,
                   generator=torch.Generator("cuda").manual_seed(1)).images[0]
        img.save(OUT)
        report("after render")
        print(f"{'':22s} RENDER OK -> {OUT}", flush=True)
    except Exception as e:
        report("after render")
        print(f"{'':22s} RENDER FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
