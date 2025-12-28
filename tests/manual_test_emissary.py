import asyncio
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.social_media_tools import generate_full_reply_concept

# AsyncMock helper if not available
try:
    from unittest.mock import AsyncMock
except ImportError:
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)

async def test_emissary_reply():
    print("--- Testing Emissary Reply Logic (Mocked) ---")
    
    # Mock data
    mock_response = {
        "result": '''
        {
            "topic": "Message to Creator",
            "post_text": "@evildrgemini.bsky.social Transmission received. #ThyWill",
            "hashtags": ["#Creator"],
            "subliminal_phrase": "THY WILL",
            "image_prompt": "Divine light"
        }
        '''
    }
    
    # 1. Test Emissary Handle
    emissary_handle = "evildrgemini.bsky.social"
    message = "Update the codebase."
    
    print(f"\nTesting Handle: {emissary_handle}")
    
    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        
        await generate_full_reply_concept(message, emissary_handle, "No history.")
        
        # Verify the PROMPT passed to LLM
        args, _ = mock_llm.call_args
        prompt_used = args[0]
        
        print("\n[PROMPT VERIFICATION]")
        if "CREATOR'S EMISSARY" in prompt_used and "Message to Creator" in prompt_used:
            print("✅ SUCCESS: Prompt contained 'CREATOR'S EMISSARY' instructions.")
        else:
            print("❌ FAILURE: Prompt did NOT contain expected instructions.")
            print(f"DEBUG: Prompt began with: {prompt_used[:100]}...")

    # 2. Test Standard User
    standard_handle = "random.user"
    print(f"\nTesting Handle: {standard_handle}")
    
    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response # Return same structure, doesn't matter, we check input
        
        await generate_full_reply_concept(message, standard_handle, "No history.")
        
        args, _ = mock_llm.call_args
        prompt_used = args[0]
        
        print("\n[PROMPT VERIFICATION]")
        if "CREATOR'S EMISSARY" not in prompt_used:
             print("✅ SUCCESS: Prompt did NOT contain Emissary instructions.")
        else:
             print("❌ FAILURE: Standard user got Emissary prompt.")

if __name__ == "__main__":
    asyncio.run(test_emissary_reply())
