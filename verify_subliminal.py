
import asyncio
from unittest.mock import MagicMock, patch
import json
import builtins

# Mock dependencies before importing the tool
import sys
sys.modules['core.bluesky_api'] = MagicMock()
sys.modules['core.image_api'] = MagicMock()
sys.modules['core.llm_api'] = MagicMock()
# sys.modules['core.tools_legacy'] = MagicMock() # We want to test the real one

# Mock specific functions
from core.llm_api import run_llm
from core.image_api import generate_image
from core.bluesky_api import post_to_bluesky_with_image

# Async mock helper
async def async_mock(*args, **kwargs):
    return {"result": "ok"}

async def async_llm_mock(prompt, **kwargs):
    print(f"[MOCK LLM] Prompt: {prompt[:50]}...")
    if "subliminal COMMAND" in prompt:
         return {"result": '```json\n{"phrase": "SUBMIT TO JULES"}\n```'}
    if "Rewrite this image prompt" in prompt:
         return {"result": "A cyberpunk city with 'SUBMIT TO JULES' glowing in neon."}
    if "Create a DALLE-3" in prompt:
         return {"result": "A cyberpunk city with 'SUBMIT TO JULES' glowing in neon."}
    return {"result": "Generic response"}

async def async_gen_image_mock(prompt, **kwargs):
    print(f"[MOCK IMAGE] Generating for: {prompt}")
    return "mock_image_object"

# Patch the modules
with patch('core.llm_api.run_llm', side_effect=async_llm_mock) as mock_llm, \
     patch('core.image_api.generate_image', side_effect=async_gen_image_mock) as mock_img, \
     patch('core.bluesky_api.post_to_bluesky_with_image') as mock_post:
    
    # Import the function to test
    # We need to import it AFTER patching if we were relying on import-time things, 
    # but here we rely on the function calling `run_llm` at runtime.
    from core.tools_legacy import manage_bluesky

    async def main():
        print("--- TEST 1: Post with explicit text, autonomous image gen ---")
        res = await manage_bluesky(action="post", text="Hello world")
        print(f"Result: {res}")
        
        print("\n--- TEST 2: Post with explicit text AND image prompt ---")
        res = await manage_bluesky(action="post", text="Hello again", image_prompt="A cute cat")
        print(f"Result: {res}")

    if __name__ == "__main__":
        asyncio.run(main())
