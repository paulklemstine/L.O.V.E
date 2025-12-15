
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from core.bluesky_api import post_to_bluesky_with_image
from PIL import Image

def test_post():
    print("Attempting to post to Bluesky...")
    
    user = os.environ.get("BLUESKY_USER")
    pwd = os.environ.get("BLUESKY_PASSWORD")
    
    if not user:
        print("ERROR: BLUESKY_USER not set in env.")
    if not pwd:
        print("ERROR: BLUESKY_PASSWORD not set in env.")
        
    print(f"User: {user}")
    
    try:
        # Create a tiny test image
        img = Image.new('RGB', (100, 100), color = 'red')
        
        response = post_to_bluesky_with_image(
            text="Debug Post: Verifying API connectivity and credentials via L.O.V.E. manual trigger.",
            image=img
        )
        print("SUCCESS!")
        print(f"Response: {response}")
    except Exception as e:
        print("FAILURE!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_post()
