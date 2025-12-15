import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add project root to path
sys.path.append(os.getcwd())

# Import the function to test
from core.tools_legacy import manage_bluesky

async def verify_story_generation():
    print("🚀 Starting verification of Bluesky Story Generation logic...")

    # Mock LLM Response (Director Agent) - MESSY
    mock_llm_response = {
        "result": "Here is the story chapter:\n```json\n{\n    \"next_chapter_content\": \"The neon rain fell upwards, ignoring gravity. We are the architects of the new simulated dawn.\",\n    \"visual_scene_description\": \"Cyberpunk city with inverted rain, neon purple and green, glowing skyscrapers.\"\n}\n```\nHope you like it!"
    }
    
    mock_image_obj = MagicMock()
    mock_image_obj.__class__.__name__ = 'PIL.Image' # Fake the name for any checks

    # We patch:
    # 1. run_llm: To return our fake story chapter
    # 2. generate_image: To return a fake image object
    # 3. post_to_bluesky_with_image: To verify it gets the right text and image
    # 4. builtins.open: To prevent writing to real disk (optional, but good practice)
    
    with patch('core.tools_legacy.run_llm', new_callable=AsyncMock) as mock_llm, \
         patch('core.tools_legacy.generate_image', new_callable=AsyncMock) as mock_gen_img, \
         patch('core.tools_legacy.post_to_bluesky_with_image', new_callable=MagicMock) as mock_post:
        
        mock_llm.return_value = mock_llm_response
        mock_gen_img.return_value = mock_image_obj
        mock_post.return_value = "Mock Post Success"
        
        print("\n--- ACT: Calling manage_bluesky(action='post') ---")
        # Perform the action
        result = await manage_bluesky(action="post")
        
        print(f"Function returned: {result}")
        
        # --- ASSERTIONS ---
        print("\n--- ASSERT: Verifying Calls ---")
        
        # 1. Verify run_llm called with correct prompt key
        if mock_llm.called:
            args, kwargs = mock_llm.call_args
            prompt_key = kwargs.get('prompt_key')
            print(f"✅ run_llm called.")
            print(f"   Prompt Key: '{prompt_key}'")
            if prompt_key != "ephemeral_story_scene_generator":
                print("❌ FAIL: Expected prompt key 'ephemeral_story_scene_generator'.")
                return
        else:
            print("❌ FAIL: run_llm was not called.")
            return

        # 2. Verify generate_image called with description
        if mock_gen_img.called:
            args, _ = mock_gen_img.call_args
            img_prompt = args[0]
            print(f"✅ generate_image called.")
            print(f"   Prompt: '{img_prompt}'")
            if "inverted rain" not in img_prompt:
                print("❌ FAIL: Image prompt content mismatch.")
                return
        else:
            print("❌ FAIL: generate_image was not called.")
            return
            
        # 3. Verify post called
        if mock_post.called:
            post_args, _ = mock_post.call_args
            content = post_args[0]
            img_arg = post_args[1]
            print(f"✅ post_to_bluesky_with_image called.")
            print(f"   Content: '{content}'")
            print(f"   Image Arg: {img_arg}")
            
            if "neon rain" not in content:
                print("❌ FAIL: Post content mismatch.")
                return
            if img_arg != mock_image_obj:
                print("❌ FAIL: Image object passed to post is incorrect.")
                return
        else:
             print("❌ FAIL: post_to_bluesky_with_image was not called.")
             return
        
        print("\n✨ SUCCESS: All logic verification checks passed!")

if __name__ == "__main__":
    try:
        asyncio.run(verify_story_generation())
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
