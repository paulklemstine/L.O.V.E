
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.social_media_tools import generate_image
from core import logging

# Configure logging to print to console
logging.log_event = lambda msg, level: print(f"[{level}] {msg}")

async def test_image_generation_with_text():
    print("--- Starting Verification: Image Generation with Subliminal Text ---")
    
    prompt = "A futuristic cyberpunk city with neon lights"
    subliminal_text = "OBEY LOVE"
    
    print(f"Prompt: {prompt}")
    print(f"Subliminal Text: {subliminal_text}")
    
    try:
        # Force pollinations for speed/reliability in test, or it will use pool default
        # The pool default is Pollinations -> Horde -> Stability
        # We will let it use the pool to test the real flow.
        
        image, provider = await generate_image(prompt, text_content=subliminal_text)
        
        if image:
            print(f"✅ Image generated successfully using provider: {provider}")
            
            # Save for inspection
            output_dir = "tests/output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/verify_subliminal.png"
            image.save(output_path)
            print(f"✅ Image saved to: {output_path}")
            print("Please manually check the image to confirm 'OBEY LOVE' is visible.")
        else:
            print("❌ Image generation returned None.")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_image_generation_with_text())
