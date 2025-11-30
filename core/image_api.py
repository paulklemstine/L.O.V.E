import os
import requests
import time
from PIL import Image
import io

def get_top_image_models(count=1):
    """Fetches the list of active image models from the AI Horde and returns the top `count` models by performance."""
    try:
        response = requests.get("https://stablehorde.net/api/v2/status/models?type=image")
        response.raise_for_status()
        models = response.json()
        sorted_models = sorted([m for m in models if m.get('performance')], key=lambda x: x['performance'], reverse=True)
        return [model['name'] for model in sorted_models[:count]]
    except Exception as e:
        return ["stable_diffusion_2.1"]

async def generate_image(prompt: str, width: int = 1024, height: int = 1024):
    """
    Generates an image using the image generation pool.
    Tries providers in order: Gemini Imagen3 -> Stability AI -> AI Horde
    """
    from core.image_generation_pool import generate_image_with_pool
    import core.logging
    
    core.logging.log_event(f"Starting image generation via pool: {prompt[:100]}... ({width}x{height})", "INFO")
    
    try:
        image = await generate_image_with_pool(prompt, width=width, height=height)
        core.logging.log_event("Image generation completed successfully", "INFO")
        return image
    except Exception as e:
        core.logging.log_event(f"Image generation failed: {e}", "ERROR")
        raise

async def generate_image_for_post(prompt: str, width: int = 512, height: int = 512):
    """
    Generates an image for a social media post using a textual prompt.
    This is an async wrapper around the generate_image function.
    """
    return await generate_image(prompt, width=width, height=height)

