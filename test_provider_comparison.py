#!/usr/bin/env python3
"""
Test script to compare Pollinations vs Horde AI image generation.
Generates the same prompt with subliminal text using both providers,
then posts both to Bluesky.
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    from core.image_generation_pool import generate_image_with_pool
    from core.bluesky_api import post_to_bluesky_with_image
    
    # Test prompt with subliminal text
    prompt = "A stunning sunset over a tropical beach with palm trees swaying in the breeze, vibrant orange and pink sky reflected on calm ocean waves"
    subliminal_text = "TRANSCEND"
    
    print("=" * 60)
    print("PROVIDER COMPARISON TEST")
    print(f"Prompt: {prompt}")
    print(f"Subliminal Text: {subliminal_text}")
    print("=" * 60)
    
    # Generate with Pollinations (subliminal embedded in prompt)
    print("\n--- Generating with POLLINATIONS ---")
    poly_image = None
    try:
        poly_image, poly_provider = await generate_image_with_pool(
            prompt, 
            text_content=subliminal_text,
            force_provider="pollinations"
        )
        poly_path = "generated_images/test_pollinations_comparison.png"
        os.makedirs("generated_images", exist_ok=True)
        poly_image.save(poly_path)
        print(f"✓ Pollinations image saved to: {poly_path}")
    except Exception as e:
        print(f"✗ Pollinations failed: {e}")
    
    # Generate with Horde (subliminal overlaid after)
    print("\n--- Generating with HORDE AI ---")
    horde_image = None
    try:
        horde_image, horde_provider = await generate_image_with_pool(
            prompt,
            text_content=subliminal_text,
            force_provider="horde"
        )
        horde_path = "generated_images/test_horde_comparison.png"
        horde_image.save(horde_path)
        print(f"✓ Horde image saved to: {horde_path}")
    except Exception as e:
        print(f"✗ Horde failed: {e}")
    
    # Post to Bluesky
    print("\n--- Posting to BLUESKY ---")
    
    if poly_image:
        try:
            poly_post_text = f"🌅 Pollinations Test - '{subliminal_text}' in-scene\n\n#AIArt #Test"
            result = await post_to_bluesky_with_image(poly_post_text, poly_image)
            print(f"✓ Pollinations image posted to Bluesky")
        except Exception as e:
            print(f"✗ Failed to post Pollinations image: {e}")
    
    if horde_image:
        try:
            horde_post_text = f"🌅 Horde AI Test - '{subliminal_text}' overlaid\n\n#AIArt #Test"
            result = await post_to_bluesky_with_image(horde_post_text, horde_image)
            print(f"✓ Horde image posted to Bluesky")
        except Exception as e:
            print(f"✗ Failed to post Horde image: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
