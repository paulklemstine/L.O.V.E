import os
import asyncio
from core.bluesky_api import post_to_bluesky_with_image

def test_hashtag_posting():
    text = "Testing programmatic hashtags #LoveAI #Python"
    print(f"Posting: {text}")
    try:
        response = post_to_bluesky_with_image(text)
        print(f"Post successful! CID: {response.cid}")
        print(f"URI: {response.uri}")
        # We can inspect the response to see if facets were created if we really wanted to, 
        # but manual verification on the site is also good.
    except Exception as e:
        print(f"Post failed: {e}")

if __name__ == "__main__":
    test_hashtag_posting()
