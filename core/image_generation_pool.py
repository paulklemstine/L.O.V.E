import os
import time
import requests
import asyncio
import aiohttp
import base64
from collections import defaultdict
from PIL import Image
import io
from PIL import Image, ImageDraw, ImageFont, ImageStat
import core.logging
import urllib.parse
import random

# --- Provider Configurations ---
GEMINI_IMAGE_MODELS = ["imagen-3.0"]  # Imagen 3
STABILITY_MODELS = ["stable-diffusion-xl-1024-v1-0"]
POLLINATIONS_MODELS = ["pollinations"]  # Primary provider
# CRAIYON_MODELS = ["craiyon"]  # DISABLED


class PollinationsCreditExhaustedException(Exception):
    """Raised when Pollinations API credits (pollen) are exhausted."""
    pass

# --- Statistics Tracking ---
def _create_default_image_model_stats():
    return {
        "total_generations": 0,
        "successful_generations": 0,
        "failed_generations": 0,
        "total_time_spent": 0.0,
        "provider": "unknown",
        "quality_score": 50.0
    }

IMAGE_MODEL_STATS = defaultdict(_create_default_image_model_stats)
IMAGE_MODEL_AVAILABILITY = {}
IMAGE_MODEL_FAILURE_COUNT = {}

# Initialize stats for known providers
for model in GEMINI_IMAGE_MODELS:
    IMAGE_MODEL_STATS[model]["provider"] = "gemini"
    IMAGE_MODEL_STATS[model]["quality_score"] = 90.0  # High quality

for model in STABILITY_MODELS:
    IMAGE_MODEL_STATS[model]["provider"] = "stability"
    IMAGE_MODEL_STATS[model]["quality_score"] = 85.0

# Pollinations - Primary provider with in-scene subliminal text
for model in POLLINATIONS_MODELS:
    IMAGE_MODEL_STATS[model]["provider"] = "pollinations"
    IMAGE_MODEL_STATS[model]["quality_score"] = 88.0  # High quality, between Gemini and Stability

# Craiyon DISABLED
# for model in CRAIYON_MODELS:
#     IMAGE_MODEL_STATS[model]["provider"] = "craiyon"
#     IMAGE_MODEL_STATS[model]["quality_score"] = 75.0


def rank_image_models():
    """
    Ranks image models based on success rate, speed, and quality.
    Returns a sorted list of model IDs.
    """
    if not IMAGE_MODEL_STATS:
        return []
    
    ranked_models = []
    for model_id, stats in IMAGE_MODEL_STATS.items():
        # Calculate success rate
        total_gens = stats["successful_generations"] + stats["failed_generations"]
        if total_gens > 0:
            success_rate = stats["successful_generations"] / total_gens
        else:
            success_rate = 0.75  # Default for untried models
        
        # Calculate speed score (generations per second)
        if stats["total_time_spent"] > 0:
            speed = stats["total_generations"] / stats["total_time_spent"]
        else:
            speed = 0
        speed_score = min(speed * 10, 100)  # Normalize
        
        # Quality score
        quality_score = stats.get("quality_score", 50.0)
        
        # Weighted final score: Quality 50%, Success 30%, Speed 20%
        final_score = (0.5 * quality_score) + (0.3 * success_rate * 100) + (0.2 * speed_score)
        
        # Boost Gemini models
        if stats.get("provider") == "gemini":
            final_score *= 1.3
        
        ranked_models.append({"model_id": model_id, "score": final_score})
    
    # Sort by score descending
    sorted_models = sorted(ranked_models, key=lambda x: x["score"], reverse=True)
    return [model["model_id"] for model in sorted_models]


# _overlay_text removed in favor of core.text_overlay_utils.overlay_text_on_image
from core.text_overlay_utils import overlay_text_on_image

def _is_image_black(image: Image.Image, threshold: int = 5) -> bool:
    """
    Checks if an image is substantially black or empty.
    Returns True if the image is considered 'black' and should be rejected.
    """
    if not image:
        return False # None is handled elsewhere
        
    try:
        # Convert to grayscale for analysis
        gray = image.convert("L")
        
        # Check 1: Pure black extrema
        extrema = gray.getextrema()
        if extrema == (0, 0):
             core.logging.log_event("Image validation failed: Image is pure black (0,0).", "WARNING")
             return True
        
        # Check 2: Average brightness
        stat = ImageStat.Stat(gray)
        avg_brightness = stat.mean[0]
        
        if avg_brightness < threshold:
             core.logging.log_event(f"Image validation failed: Image is too dark. Avg brightness: {avg_brightness:.2f} < {threshold}", "WARNING")
             return True
             
        return False
    except Exception as e:
        core.logging.log_event(f"Image validation error: {e}", "WARNING")
        return False



async def _generate_with_gemini_imagen(prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
    """Generate image using Gemini Imagen3 via Gemini API"""
    model_id = "imagen-3.0-generate-001"
    start_time = time.time()
    
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        core.logging.log_event(f"Attempting image generation with Gemini Imagen3: {prompt[:100]}...", "INFO")
        
        # Using the predict endpoint for Imagen 3 on Generative Language API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict"
        
        # Determine aspect ratio from width/height
        aspect_ratio = "1:1"
        if width > height:
            aspect_ratio = "16:9"
        elif height > width:
            aspect_ratio = "9:16"

        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        
        # Construct payload for 'predict' method
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, params=params, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract image data from 'predictions'
                # Typical response: {"predictions": [{"bytesBase64Encoded": "..."}]}
                if "predictions" in data and len(data["predictions"]) > 0:
                    prediction = data["predictions"][0]
                    
                    if "bytesBase64Encoded" in prediction:
                        image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                    elif "mimeType" in prediction and "bytesBase64Encoded" in prediction:
                        # Sometimes wrapped differently
                        image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                    else:
                        # Fallback for potential variation
                        core.logging.log_event(f"Unexpected prediction format: {prediction.keys()}", "WARNING")
                        if "image" in prediction: # older format?
                             image_bytes = base64.b64decode(prediction["image"])
                        else:
                            raise Exception(f"Could not find image data in prediction: {prediction.keys()}")
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Update stats
                    elapsed = time.time() - start_time
                    IMAGE_MODEL_STATS[model_id]["successful_generations"] += 1
                    IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
                    IMAGE_MODEL_STATS[model_id]["total_time_spent"] += elapsed
                    IMAGE_MODEL_FAILURE_COUNT[model_id] = 0
                    
                    core.logging.log_event(f"Gemini Imagen3 generation successful in {elapsed:.2f}s", "INFO")
                    return image
                else:
                    raise Exception(f"No predictions in response: {data.keys()}")
                    
    except Exception as e:
        elapsed = time.time() - start_time
        IMAGE_MODEL_STATS[model_id]["failed_generations"] += 1
        IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
        
        failure_count = IMAGE_MODEL_FAILURE_COUNT.get(model_id, 0) + 1
        IMAGE_MODEL_FAILURE_COUNT[model_id] = failure_count
        
        # Apply exponential backoff cooldown
        cooldown = 60 * (2 ** (failure_count - 1))
        IMAGE_MODEL_AVAILABILITY[model_id] = time.time() + cooldown
        
        core.logging.log_event(f"Gemini Imagen3 failed: {e}. Cooldown: {cooldown}s", "WARNING")
        raise


async def _generate_with_stability(prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
    """Generate image using Stability AI API"""
    model_id = "stable-diffusion-xl-1024-v1-0"
    start_time = time.time()
    
    try:
        api_key = os.environ.get("STABILITY_API_KEY")
        if not api_key:
            raise ValueError("STABILITY_API_KEY not set")
        
        core.logging.log_event(f"Attempting image generation with Stability AI: {prompt[:100]}...", "INFO")
        
        url = f"https://api.stability.ai/v1/generation/{model_id}/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": 30
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=120) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Stability AI error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract base64 image
                if "artifacts" in data and len(data["artifacts"]) > 0:
                    image_b64 = data["artifacts"][0]["base64"]
                    image_bytes = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Update stats
                    elapsed = time.time() - start_time
                    IMAGE_MODEL_STATS[model_id]["successful_generations"] += 1
                    IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
                    IMAGE_MODEL_STATS[model_id]["total_time_spent"] += elapsed
                    IMAGE_MODEL_FAILURE_COUNT[model_id] = 0
                    
                    core.logging.log_event(f"Stability AI generation successful in {elapsed:.2f}s", "INFO")
                    return image
                else:
                    raise Exception("No artifacts in response")
                    
    except Exception as e:
        elapsed = time.time() - start_time
        IMAGE_MODEL_STATS[model_id]["failed_generations"] += 1
        IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
        
        failure_count = IMAGE_MODEL_FAILURE_COUNT.get(model_id, 0) + 1
        IMAGE_MODEL_FAILURE_COUNT[model_id] = failure_count
        
        cooldown = 60 * (2 ** (failure_count - 1))
        IMAGE_MODEL_AVAILABILITY[model_id] = time.time() + cooldown
        
        core.logging.log_event(f"Stability AI failed: {e}. Cooldown: {cooldown}s", "WARNING")
        raise


async def _generate_with_horde(prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
    """Generate image using AI Horde (existing logic from image_api.py)"""
    from core.image_api import get_top_image_models
    
    model_id = "ai_horde"
    start_time = time.time()
    
    try:
        core.logging.log_event(f"Attempting image generation with AI Horde: {prompt[:100]}...", "INFO")
        
        api_key = os.environ.get("STABLE_HORDE", "0000000000")
        headers = {"apikey": api_key, "Content-Type": "application/json"}
        
        top_models = get_top_image_models()
        if not top_models:
            raise Exception("Could not fetch any image models from AI Horde")
        
        core.logging.log_event(f"AI Horde selected models: {top_models}", "INFO")

        payload = {
            "prompt": prompt,
            "params": {
                "n": 1,
                "width": width,
                "height": height,
                "steps": 30,
                "cfg_scale": 7.0,
                "sampler_name": "k_dpmpp_2m",
                "karras": True,
                "tiling": False,
                "hires_fix": False
            },
            "nsfw": True,
            "censor_nsfw": False,
            "trusted_workers": False,
            "models": top_models
        }
        
        # Submit the request
        api_url = "https://stablehorde.net/api/v2/generate/async"
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers, timeout=30) as response:
                response.raise_for_status()
                job = await response.json()
                job_id = job["id"]
                core.logging.log_event(f"AI Horde job submitted: {job_id}", "INFO")
            
            # Poll for result
            check_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
            for attempt in range(60):  # Poll for 10 minutes
                await asyncio.sleep(10)
                async with session.get(check_url, headers=headers, timeout=10) as check_response:
                    check_response.raise_for_status()
                    status = await check_response.json()
                    
                    if attempt % 6 == 0:
                        core.logging.log_event(f"AI Horde in progress... (attempt {attempt+1}/60)", "INFO")
                    
                    if status["done"]:
                        img_url = status["generations"][0]["img"]
                        async with session.get(img_url, timeout=30) as img_response:
                            img_response.raise_for_status()
                            image_bytes = await img_response.read()
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Update stats
                            elapsed = time.time() - start_time
                            IMAGE_MODEL_STATS[model_id]["successful_generations"] += 1
                            IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
                            IMAGE_MODEL_STATS[model_id]["total_time_spent"] += elapsed
                            IMAGE_MODEL_FAILURE_COUNT[model_id] = 0
                            
                            core.logging.log_event(f"AI Horde generation successful in {elapsed:.2f}s", "INFO")
                            return image
            
            raise Exception("AI Horde job timed out after 10 minutes")
            
    except Exception as e:
        elapsed = time.time() - start_time
        IMAGE_MODEL_STATS[model_id]["failed_generations"] += 1
        IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
        
        failure_count = IMAGE_MODEL_FAILURE_COUNT.get(model_id, 0) + 1
        IMAGE_MODEL_FAILURE_COUNT[model_id] = failure_count
        
        cooldown = 60 * (2 ** (failure_count - 1))
        IMAGE_MODEL_AVAILABILITY[model_id] = time.time() + cooldown
        
        core.logging.log_event(f"AI Horde failed: {e}. Cooldown: {cooldown}s", "WARNING")
        raise


async def _generate_with_pollinations(prompt: str, width: int = 1024, height: int = 1024, subliminal_text: str = None) -> Image.Image:
    """
    Generate image using Pollinations.ai with authenticated API.
    
    Args:
        prompt: Text description of image to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        subliminal_text: Optional text to embed IN the image generation prompt.
                        This instructs the LLM to render the text as an in-scene element
                        (graffiti, neon sign, holographic display, etc.)
    """
    model_id = "pollinations"
    start_time = time.time()
    
    try:
        api_key = os.environ.get("POLLINATIONS_API_KEY")
        if not api_key:
            raise ValueError("POLLINATIONS_API_KEY not set")
        
        core.logging.log_event(f"Attempting image generation with Pollinations: {prompt[:100]}...", "INFO")
        
        # Embed subliminal text directly into the prompt for in-scene rendering
        enhanced_prompt = prompt
        if subliminal_text:
            # Add specific instructions to render the subliminal/manipulative text in-scene
            subliminal_instruction = (
                f" CRITICAL: Prominently render the exact text \"{subliminal_text}\" "
                f"as a visible in-scene element - as graffiti on a wall, "
                f"neon signage, holographic floating text, LED display, "
                f"tattoo on skin, or light projection in the environment. "
                f"The text must be clearly legible and integrated naturally into the scene."
            )
            enhanced_prompt = prompt + subliminal_instruction
            core.logging.log_event(f"Pollinations prompt enhanced with subliminal: '{subliminal_text}'", "INFO")
        
        # URL Encode the enhanced prompt
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        
        # Generate a random seed
        seed = random.randint(0, 2147483647)  # Max seed per API docs
        
        # Use gen.pollinations.ai with API key authentication
        # Models: flux (default), turbo, gptimage, kontext, seedream
        model = "flux"  # High quality model
        url = f"https://gen.pollinations.ai/image/{encoded_prompt}?model={model}&width={width}&height={height}&seed={seed}&safe=false&enhance=true"
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=120) as response:
                # Check for credit exhaustion (401 Unauthorized or 402 Payment Required)
                if response.status in (401, 402):
                    error_text = await response.text()
                    raise PollinationsCreditExhaustedException(f"Pollinations credits exhausted (HTTP {response.status}): {error_text}")
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Pollinations API error {response.status}: {error_text}")
                
                image_bytes = await response.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Update stats
                elapsed = time.time() - start_time
                IMAGE_MODEL_STATS[model_id]["successful_generations"] += 1
                IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
                IMAGE_MODEL_STATS[model_id]["total_time_spent"] += elapsed
                IMAGE_MODEL_FAILURE_COUNT[model_id] = 0
                
                core.logging.log_event(f"Pollinations generation successful in {elapsed:.2f}s", "INFO")
                return image
                
    except PollinationsCreditExhaustedException:
        # Re-raise credit exhaustion to be handled specially by the pool
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        IMAGE_MODEL_STATS[model_id]["failed_generations"] += 1
        IMAGE_MODEL_STATS[model_id]["total_generations"] += 1
        
        failure_count = IMAGE_MODEL_FAILURE_COUNT.get(model_id, 0) + 1
        IMAGE_MODEL_FAILURE_COUNT[model_id] = failure_count
        
        cooldown = 60 * (2 ** (failure_count - 1))
        IMAGE_MODEL_AVAILABILITY[model_id] = time.time() + cooldown
        
        core.logging.log_event(f"Pollinations failed: {e}. Cooldown: {cooldown}s", "WARNING")
        raise


# _generate_with_craiyon DISABLED - Craiyon removed as image provider

from typing import Tuple

async def generate_image_with_pool(prompt: str, width: int = 1024, height: int = 1024, force_provider=None, text_content: str = None, overlay_position: str = None) -> Tuple[Image.Image, str]:
    """
    Main entry point for image generation using the pool.
    Tries providers in order based on ranking, with automatic fallback.
    
    Args:
        prompt: Text description of the image to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        force_provider: Optional provider to force ("gemini", "stability", "horde")
        text_content: Optional text to embed/overlay
    
    Returns:
        PIL Image object
    """
    core.logging.log_event(f"Starting image generation with pool: {prompt[:100]}... ({width}x{height}) [Text: {text_content}]", "INFO")
    
    # Define provider functions
    providers = {
        "gemini": _generate_with_gemini_imagen,
        "stability": _generate_with_stability,
        "horde": _generate_with_horde,
        "pollinations": _generate_with_pollinations,  # DISABLED in provider_order
        # "craiyon": _generate_with_craiyon  # DISABLED
    }
    
    if force_provider:
        if force_provider not in providers:
            raise ValueError(f"Unknown provider: {force_provider}")
        provider_order = [force_provider]
        core.logging.log_event(f"Forcing image generation with provider: {force_provider}", "INFO")
    else:
        # Try providers in order: Pollinations (with in-scene text) -> Horde (with overlay)
        # Pollinations embeds subliminal text IN the generation prompt for in-scene rendering
        # Horde falls back to manual text overlay post-generation
        provider_order = ["pollinations", "horde"]
    
    last_exception = None
    
    for provider_name in provider_order:
        # Check if provider is available (not on cooldown)
        provider_key = f"{provider_name}_provider"
        if time.time() < IMAGE_MODEL_AVAILABILITY.get(provider_key, 0):
            cooldown_remaining = IMAGE_MODEL_AVAILABILITY[provider_key] - time.time()
            core.logging.log_event(f"Provider {provider_name} is on cooldown for {cooldown_remaining:.0f}s", "INFO")
            continue
            
        # --- Pre-flight Checks ---
        if provider_name == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            continue
        if provider_name == "stability" and not os.environ.get("STABILITY_API_KEY"):
            continue
        if provider_name == "pollinations" and not os.environ.get("POLLINATIONS_API_KEY"):
            continue
        
        try:
            core.logging.log_event(f"Trying image generation with provider: {provider_name}", "INFO")
            
            # --- PRE-GENERATION PROVIDER LOGIC ---
            current_prompt = prompt
            
            # For Pollinations: Pass subliminal text to be embedded in prompt (rendered in-scene by LLM)
            # For other providers: Generate clean image, apply overlay after
            if provider_name == "pollinations":
                # Pollinations gets subliminal text embedded in prompt for in-scene rendering
                image = await providers[provider_name](current_prompt, width=width, height=height, subliminal_text=text_content)
                # No overlay needed - text rendered in-scene by the model
                manual_overlay_text = None
            else:
                # Non-Pollinations: generate without subliminal, apply overlay after
                image = await providers[provider_name](current_prompt, width=width, height=height)
                manual_overlay_text = text_content  # Will be overlaid after generation
            
            if image:
                # --- VALIDATION ---
                # Check for black/blank images
                if _is_image_black(image):
                    core.logging.log_event(f"Provider {provider_name} returned a black/blank image. Skipping.", "WARNING")
                    # Treat as failure -> Cooldown?
                    # For now just skip to next provider to avoid punishing transient errors too hard, 
                    # but we should probably record it.
                    last_exception = Exception("Generated image was black/blank")
                    continue
                # ------------------

                # --- POST-GENERATION OVERLAY LOGIC ---
                # ALWAYS apply subliminal text overlay if text is provided
                if manual_overlay_text:
                    # Randomize position between top, center, bottom
                    position_choices = ["top", "center", "bottom"]
                    selected_position = overlay_position or random.choice(position_choices)
                    
                    core.logging.log_event(f"Applying manual text overlay: '{manual_overlay_text}' at position: {selected_position}", "INFO")
                    try:
                        image = overlay_text_on_image(image, manual_overlay_text, position=selected_position, style="subliminal")
                    except Exception as e:
                         core.logging.log_event(f"Failed to overlay text: {e}", "ERROR")
                # -------------------------------------
                
                core.logging.log_event(f"Image generation successful with provider: {provider_name}", "INFO")
                core.logging.log_event(f"Image generation successful with provider: {provider_name}", "INFO")
                return image, provider_name
            
        except PollinationsCreditExhaustedException as e:
            # Credit exhaustion - immediately fall back to next provider, set long cooldown
            core.logging.log_event(f"Pollinations credits exhausted, falling back to next provider: {e}", "WARNING")
            # Set a 24-hour cooldown on Pollinations when credits are exhausted
            IMAGE_MODEL_AVAILABILITY["pollinations_provider"] = time.time() + (24 * 60 * 60)
            last_exception = e
            continue
        except Exception as e:
            last_exception = e
            core.logging.log_event(f"Provider {provider_name} failed: {e}", "WARNING")
            continue
    
    # All providers failed
    error_msg = f"All image generation providers failed. Last error: {last_exception}"
    core.logging.log_event(error_msg, "ERROR")
    raise Exception(error_msg)
