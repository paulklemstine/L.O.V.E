#!/usr/bin/env python3
"""Local image generation via diffusers (SDXL), replacing Pollinations /image calls.

Usage:
  generate_image.py --prompt "..." [--negative "..."] [--w 1024] [--h 1024]
                    [--seed N] [--steps 30] [--out /path/out.png]

Designed for a 6GB GPU: fp16 + attention slicing + VAE tiling + model CPU offload.
"""
import argparse, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_prompt import encode_long_prompt

# Model weights (~25GB) and the venv live outside the repo; only the scripts are
# versioned here. Override with LOVE_AI_HOME if the toolchain lives elsewhere.
AI_HOME = os.environ.get("LOVE_AI_HOME", os.path.join(os.path.expanduser("~"), "ai"))
MODEL_PATHS = [
    os.path.join(AI_HOME, name)
    for name in ("sdxl", "leosam", "realvis")
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="blurry, low quality, deformed, watermark, text artifacts")
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    model_path = random.choice(MODEL_PATHS)
    print(f"model: {model_path}", file=sys.stderr)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    prompt = args.prompt
    negative = args.negative

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    gen = torch.Generator("cuda").manual_seed(seed)
    # Chunked encoding: CLIP truncates past 77 tokens, which silently drops the
    # palette/composition tail of these prompts.
    pe, npe, pp, npp = encode_long_prompt(prompt, negative, pipe, pipe._execution_device)
    image = pipe(
        prompt_embeds=pe,
        negative_prompt_embeds=npe,
        pooled_prompt_embeds=pp,
        negative_pooled_prompt_embeds=npp,
        width=args.w,
        height=args.h,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).images[0]
    image.save(args.out)
    print(f"saved {args.out} seed={seed}")

if __name__ == "__main__":
    sys.exit(main())
