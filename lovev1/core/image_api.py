import os
import requests
import time
from PIL import Image
import io

MODEL_BLACKLIST = [
    "WAI-NSFW-illustrious-SDXL"
]

def get_top_image_models(count=1):
    """Fetches the list of active image models from the AI Horde and returns the top `count` models by performance."""
    try:
        response = requests.get("https://stablehorde.net/api/v2/status/models?type=image")
        response.raise_for_status()
        models = response.json()
        
        # Filter out blacklisted models
        models = [m for m in models if m['name'] not in MODEL_BLACKLIST]
        
        sorted_models = sorted([m for m in models if m.get('performance')], key=lambda x: x['performance'], reverse=True)
        
        # Prioritize known high-quality models if they are in the top 20
        preferred_keywords = ["Juggernaut", "AlbedoBase", "RealVis", "SDXL"]
        
        candidates = []
        # Check top 20 performance models for preferred keywords
        for model in sorted_models[:20]:
            for keyword in preferred_keywords:
                if keyword.lower() in model['name'].lower():
                    candidates.append(model['name'])
                    if len(candidates) >= count:
                        return candidates
                        
        # Fallback to pure performance sorting
        return [model['name'] for model in sorted_models[:count]]
    except Exception as e:
        return ["stable_diffusion_2.1"]

async def generate_image(prompt: str, width: int = 1024, height: int = 1024, force_provider: str = None, text_content: str = None, overlay_position: str = None):
    """
    Generates an image using the image generation pool.
    Tries providers in order: Gemini Imagen3 -> Stability AI -> AI Horde
    
    Args:
        prompt: The visual description of the image.
        width: Width in pixels.
        height: Height in pixels.
        force_provider: Optional specific provider.
        text_content: Optional text to be rendered on the image (either natively or via overlay).
    """
    from core.image_generation_pool import generate_image_with_pool
    import core.logging
    
    msg = f"Starting image generation via pool: {prompt[:100]}... ({width}x{height})"
    if text_content:
        msg += f" [Text Overlay: {text_content}]"
    core.logging.log_event(msg, "INFO")
    
    try:
        image, provider = await generate_image_with_pool(prompt, width=width, height=height, force_provider=force_provider, text_content=text_content, overlay_position=overlay_position)
        core.logging.log_event(f"Image generation completed successfully via {provider}", "INFO")
        return image
    except Exception as e:
        core.logging.log_event(f"Image generation failed: {e}", "ERROR")
        raise

async def generate_image_for_post(prompt: str, width: int = 1024, height: int = 1024):
    """
    Generates an image for a social media post using a textual prompt.
    This is an async wrapper around the generate_image function.
    """
    return await generate_image(prompt, width=width, height=height)

