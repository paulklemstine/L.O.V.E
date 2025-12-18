
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
    print("Attempting to post to Bluesky (Text Only)...")
    
    user = os.environ.get("BLUESKY_USER")
    if not user:
        print("ERROR: BLUESKY_USER not set.")
        return

    try:
        # Test 1: Text Only
        print("--- Test 1: Text Only ---")
        response = post_to_bluesky_with_image(
            text="Debug Post: Text Only verification via L.O.V.E. #debug",
            image=None
        )
        print("SUCCESS (Text Only)!")
        print(f"Response: {response}")

    except Exception as e:
        print("FAILURE (Text Only)!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_post()
