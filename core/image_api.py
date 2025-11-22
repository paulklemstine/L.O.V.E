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

def generate_image(prompt: str):
    """
    Generates an image using the AI Horde API.
    """
    import core.logging
    core.logging.log_event(f"Starting AI Horde image generation with prompt: {prompt[:100]}...", "INFO")
    
    api_key = os.environ.get("STABLE_HORDE", "0000000000")
    headers = {"apikey": api_key, "Content-Type": "application/json"}

    top_models = get_top_image_models()
    if not top_models:
        core.logging.log_event("Could not fetch any image models from AI Horde.", "ERROR")
        raise Exception("Could not fetch any image models from AI Horde.")

    payload = {
        "prompt": prompt,
        "params": {"n": 1},
        "models": top_models
    }

    # Submit the request
    api_url = "https://stablehorde.net/api/v2/generate/async"
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        job_id = response.json()["id"]
        core.logging.log_event(f"AI Horde image generation job submitted. Job ID: {job_id}", "INFO")
    except Exception as e:
        core.logging.log_event(f"Failed to submit AI Horde image generation request: {e}", "ERROR")
        raise

    # Poll for the result
    check_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
    for attempt in range(60):  # Poll for 10 minutes
        time.sleep(10)
        try:
            check_response = requests.get(check_url, headers=headers, timeout=10)
            check_response.raise_for_status()
            status = check_response.json()
            
            if attempt % 6 == 0:  # Log every minute
                core.logging.log_event(f"AI Horde image generation in progress... (attempt {attempt+1}/60)", "INFO")
            
            if status["done"]:
                img_url = status["generations"][0]["img"]
                img_response = requests.get(img_url, timeout=30)
                img_response.raise_for_status()
                core.logging.log_event(f"AI Horde image generation completed successfully.", "INFO")
                return Image.open(io.BytesIO(img_response.content))
        except Exception as e:
            core.logging.log_event(f"Error checking AI Horde status (attempt {attempt+1}): {e}", "WARNING")
            if attempt >= 5:  # Give up after multiple failures
                raise

    core.logging.log_event("AI Horde image generation job timed out after 10 minutes.", "ERROR")
    raise Exception("AI Horde image generation job timed out.")


def generate_image_for_post(prompt: str):
    """
    Generates an image for a social media post using a textual prompt.
    This is a wrapper around the more generic generate_image function.
    """
    return generate_image(prompt)
