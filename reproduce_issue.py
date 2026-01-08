
import asyncio
import os
import sys

# Add the current directory to sys.path
sys.path.append(os.getcwd())

from core.image_generation_pool import generate_image_with_pool
from core.text_overlay_utils import overlay_text_on_image
from PIL import Image

async def test_generation():
    print("Testing image generation...")
    prompt = "A beautiful sunrise over a cyberpunk city, neon lights, 8k masterpiece"
    text_content = "WAKE UP"
    
    try:
        # Default pool usage
        image, provider = await generate_image_with_pool(prompt, width=512, height=512, text_content=text_content)
        
        if image:
            print(f"Image generated via {provider}")
            image.save("test_output.png")
            print("Saved to test_output.png")
            
            # Check if image is all black
            extrema = image.convert("L").getextrema()
            print(f"Image extrema (grayscale): {extrema}")
            if extrema == (0, 0):
                print("FAILURE: Image is all black!")
            else:
                print("SUCCESS: Image content detected.")
        else:
            print("No image returned.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_generation())
