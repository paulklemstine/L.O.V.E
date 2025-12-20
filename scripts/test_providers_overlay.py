
import asyncio
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.image_generation_pool import generate_image_with_pool
import core.logging

# Setup basic logging to console
import logging
logging.basicConfig(level=logging.INFO)

async def test_providers():
    print("Starting Provider Overlay Test Run 3...")
    
    prompt = "A futuristic cyberpunk city with neon billboards and rainy streets, kawaii style"
    
    # 1. Test Pollinations
    print("\n--- Testing Pollinations ---")
    try:
        # Code now handles embedding this text into the prompt for Pollinations
        img_poly = await generate_image_with_pool(
            prompt, 
            text_content="POLLINATIONS V3", 
            force_provider="pollinations"
        )
        if img_poly:
            path = "generated_images/test_pollinations_overlay_v3.png"
            img_poly.save(path)
            print(f"Saved Pollinations image to: {path}")
        else:
            print("Pollinations generation failed.")
    except Exception as e:
        print(f"Pollinations error: {e}")

    # 2. Test Horde
    print("\n--- Testing Horde ---")
    try:
        # Code now handles using this as a manual overlay for Horde
        img_horde = await generate_image_with_pool(
            prompt, 
            text_content="HORDE V3", 
            force_provider="horde"
        )
        if img_horde:
            path = "generated_images/test_horde_overlay_v3.png"
            img_horde.save(path)
            print(f"Saved Horde image to: {path}")
        else:
            print("Horde generation failed.")
    except Exception as e:
        print(f"Horde error: {e}")

if __name__ == "__main__":
    # Ensure generated_images dir exists
    os.makedirs("generated_images", exist_ok=True)
    
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_providers())
