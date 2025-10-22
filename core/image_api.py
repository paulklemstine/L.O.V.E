import os
import requests
import time
from PIL import Image
import io

def get_top_image_models(count=1):
    """Fetches the list of active image models from the AI Horde and returns the top `count` models by performance."""
    try:
        response = requests.get("https://aihorde.net/api/v2/models?type=image")
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
    api_key = os.environ.get("STABLE_HORDE", "0000000000")
    headers = {"apikey": api_key, "Content-Type": "application/json"}

    top_models = get_top_image_models()
    if not top_models:
        raise Exception("Could not fetch any image models from AI Horde.")

    payload = {
        "prompt": prompt,
        "params": {"n": 1},
        "models": top_models
    }

    # Submit the request
    api_url = "https://aihorde.net/api/v2/generate/image/async"
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    job_id = response.json()["id"]

    # Poll for the result
    check_url = f"https://aihorde.net/api/v2/generate/image/status/{job_id}"
    for _ in range(30):  # Poll for 5 minutes
        time.sleep(10)
        check_response = requests.get(check_url, headers=headers)
        check_response.raise_for_status()
        status = check_response.json()
        if status["done"]:
            img_url = status["generations"][0]["img"]
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            return Image.open(io.BytesIO(img_response.content))

    raise Exception("AI Horde image generation job timed out.")
