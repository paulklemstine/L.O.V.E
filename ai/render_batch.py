#!/usr/bin/env python3
"""Render a batch of images in one process (model loads once).

Input: JSON file with a list of {prompt, negative, w, h, out}.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_prompt import encode_long_prompt

# Model weights (~25GB) and the venv live outside the repo; only the scripts are
# versioned here. Override with LOVE_AI_HOME if the toolchain lives elsewhere.
AI_HOME = os.environ.get("LOVE_AI_HOME", os.path.join(os.path.expanduser("~"), "ai"))
MODEL_PATHS = [
    os.path.join(AI_HOME, name)
    for name in ("sdxl", "leosam", "realvis")
]

def load_pipe(path, torch, cls, sched):
    pipe = cls.from_pretrained(
        path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = sched.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    return pipe

def main():
    jobs = json.load(open(sys.argv[1]))
    import torch
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    # assign a random model per job, then render grouped so each model loads once
    for job in jobs:
        job["model"] = random.choice(MODEL_PATHS)
    loaded = {}

    for i, job in enumerate(jobs, 1):
        m = job["model"]
        if m not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            print(f"[batch] loading model {m}", flush=True)
            loaded[m] = load_pipe(m, torch, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler)
        pipe = loaded[m]
        seed = job.get("seed") or random.randint(0, 2**31 - 1)
        gen = torch.Generator("cuda").manual_seed(seed)
        prompt, negative = job["prompt"], job.get("negative")
        # Prompts routinely exceed CLIP's 77-token window; chunked encoding
        # keeps the tail (palette, composition) instead of silently dropping it.
        ntok = len(pipe.tokenizer(prompt, truncation=False,
                                  add_special_tokens=False)["input_ids"])
        device = pipe._execution_device
        pe, npe, pp, npp = encode_long_prompt(prompt, negative, pipe, device)
        # The text encoders sit on the GPU during encoding; release them before
        # the UNet is loaded, or the two together exceed a 6GB card.
        torch.cuda.empty_cache()
        print(f"[batch {i}/{len(jobs)}] {m.split('/')[-1]} {job['out']} seed={seed} "
              f"tokens={ntok} embeds={tuple(pe.shape)}", flush=True)
        image = pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=npe,
            pooled_prompt_embeds=pp,
            negative_pooled_prompt_embeds=npp,
            width=job["w"],
            height=job["h"],
            num_inference_steps=job.get("steps", 28),
            guidance_scale=7.0,
            generator=gen,
        ).images[0]
        image.save(job["out"])
        print(f"[batch {i}/{len(jobs)}] saved", flush=True)
    print("BATCH_DONE", flush=True)

if __name__ == "__main__":
    main()
