#!/usr/bin/env python3
"""
Generate Transparent L.O.V.E. Logos using Pollinations AI

Uses the Pollinations API with gptimage model and transparent=true
to generate PNG logos with alpha channel transparency.
"""
import os
import sys
import shutil
import asyncio
import aiohttp
import urllib.parse
import random
from PIL import Image
from io import BytesIO
from datetime import datetime

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LOGOS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logos")
BACKUP_DIR = os.path.join(LOGOS_DIR, "backup")

# Logo themes and prompts - designed for transparent backgrounds
LOGO_PROMPTS = {
    "logo_cyberpunk": (
        "Minimalist glowing neon heart icon logo with circuit board traces inside, "
        "cyberpunk style, electric blue and hot pink neon glow, "
        "clean vector design suitable for watermark, no background, "
        "isolated on transparent, high contrast edges"
    ),
    "logo_ethereal": (
        "Ethereal glowing heart icon with soft luminescent aura, "
        "mystical crystal heart shape emanating gentle light rays, "
        "dreamy pastel colors, fantasy style logo, "
        "isolated icon on transparent background, watermark suitable"
    ),
    "logo_minimalist": (
        "Ultra clean minimalist heart shape icon, "
        "simple elegant geometric lines, modern flat design, "
        "single color with subtle gradient, professional logo style, "
        "isolated on transparent background, watermark quality"
    ),
    "logo_organic": (
        "Organic flowing heart shape with botanical vine elements, "
        "natural curves and leaf patterns integrated, earthy tones, "
        "hand-drawn style logo icon, nature inspired, "
        "isolated on transparent background"
    ),
    "logo_vaporwave": (
        "Retro vaporwave heart icon with 80s aesthetic, "
        "chrome reflective surface, pink and cyan color scheme, "
        "subtle grid lines, synthwave style logo, "
        "isolated on transparent background, clean edges"
    ),
}

# Generation parameters
WIDTH = 512  # Icon size
HEIGHT = 512
MODEL = "gptimage"  # Only model supporting transparency


async def generate_transparent_logo(prompt: str, filename: str, api_key: str) -> bool:
    """
    Generate a single transparent logo using Pollinations API.
    
    Args:
        prompt: Text prompt for logo generation
        filename: Output filename (without path)
        api_key: Pollinations API key
    
    Returns:
        True if successful, False otherwise
    """
    seed = random.randint(0, 2147483647)
    encoded_prompt = urllib.parse.quote(prompt)
    
    # Build URL with transparent=true
    url = (
        f"https://gen.pollinations.ai/image/{encoded_prompt}"
        f"?model={MODEL}&width={WIDTH}&height={HEIGHT}"
        f"&seed={seed}&transparent=true&safe=false&enhance=true"
    )
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"  Generating: {filename}")
    print(f"  Prompt: {prompt[:80]}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=120) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"  ✗ Failed ({response.status}): {error_text[:100]}")
                    return False
                
                image_bytes = await response.read()
                image = Image.open(BytesIO(image_bytes))
                
                # Verify image has alpha channel
                if image.mode != 'RGBA':
                    print(f"  ⚠ Converting to RGBA (was {image.mode})")
                    image = image.convert('RGBA')
                
                # Check for actual transparency
                alpha = image.split()[-1]
                alpha_extrema = alpha.getextrema()
                has_transparency = alpha_extrema[0] < 255
                
                if has_transparency:
                    print(f"  ✓ Transparency verified (alpha range: {alpha_extrema})")
                else:
                    print(f"  ⚠ No transparent pixels detected")
                
                # Save as PNG to preserve transparency
                output_path = os.path.join(LOGOS_DIR, filename)
                image.save(output_path, "PNG")
                print(f"  ✓ Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
                return True
                
    except asyncio.TimeoutError:
        print(f"  ✗ Timeout generating {filename}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def backup_existing_logos():
    """Backup existing logos to backup directory."""
    if not os.path.exists(LOGOS_DIR):
        print(f"Logos directory not found: {LOGOS_DIR}")
        return
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backed_up = 0
    for filename in os.listdir(LOGOS_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('.'):
            src = os.path.join(LOGOS_DIR, filename)
            dst = os.path.join(BACKUP_DIR, f"{timestamp}_{filename}")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                backed_up += 1
    
    print(f"Backed up {backed_up} logos to {BACKUP_DIR}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("L.O.V.E. Transparent Logo Generator")
    print("Using Pollinations AI with gptimage model")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        print("ERROR: POLLINATIONS_API_KEY not set in .env")
        sys.exit(1)
    print(f"API Key: {api_key[:8]}...")
    
    # Backup existing logos
    print("\n--- Backing up existing logos ---")
    backup_existing_logos()
    
    # Generate new transparent logos
    print("\n--- Generating transparent logos ---")
    
    success_count = 0
    fail_count = 0
    
    for name, prompt in LOGO_PROMPTS.items():
        filename = f"{name}.png"
        print(f"\n[{success_count + fail_count + 1}/{len(LOGO_PROMPTS)}] {name}")
        
        if await generate_transparent_logo(prompt, filename, api_key):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"COMPLETE: {success_count} successful, {fail_count} failed")
    print("=" * 60)
    
    if fail_count > 0:
        print("\nNote: Failed logos can be regenerated by running this script again.")
    
    return success_count > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
