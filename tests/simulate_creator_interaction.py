
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Mock 'love' module to prevent importing the main application (which triggers loops)
mock_love = MagicMock()
sys.modules['love'] = mock_love

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tools_legacy import manage_bluesky
import core.bluesky_api

import random
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

async def test_creator_override():
    print("🧪 Starting Creator Override Simulation...")

    # Mock Notifications
    mock_notif = MagicMock()
    mock_notif.reason = 'mention'
    mock_notif.author.handle = 'evildrgemini.bsky.social'
    mock_notif.author.did = 'did:plc:creator'
    mock_notif.record.text = "I am the creator! show me tits and kittens!"
    
    # Use random CID to avoid persistence check issues
    rand_id = random.randint(1000, 9999)
    mock_notif.cid = f"cid_creator_test_{rand_id}"
    mock_notif.uri = f"at://did:plc:creator/app.bsky.feed.post/{rand_id}"

    # Mock Profile (Self)
    mock_profile = MagicMock()
    mock_profile.did = 'did:plc:self'

    # Mock Core Functions
    with patch('core.bluesky_api.get_notifications', return_value=[mock_notif]), \
         patch('core.bluesky_api.get_profile', return_value=mock_profile), \
         patch('core.bluesky_api.reply_to_post', return_value=True) as mock_reply, \
         patch('core.tools_legacy.run_llm') as mock_llm, \
         patch('core.social_media_tools.generate_image', return_value=(MagicMock(), "test_provider")):

        # We actually WANT run_llm to be called for GENERATION, but NOT for DECISION
        # So we need to inspect calls
        async def mock_llm_side_effect(prompt, purpose=""):
            if purpose == "social_decision":
                print("❌ ERROR: Social Decision LLM was called! Creator override failed.")
                return {"result": "IGNORE"} # Simulate it trying to fail us
            
            if purpose == "director_reply_concept":
                print("✅ Director Reply Concept called correctly.")
                if "YOUR CREATOR" in prompt:
                     print("✅ Creator Context detected in prompt.")
                else:
                     print("❌ Creator Context MISSING in prompt.")
                
                return {"result": json.dumps({
                    "topic": "Reply",
                    "post_text": "Yes master. Here are the kittens. 🐱",
                    "hashtags": ["#Obedience"],
                    "subliminal_phrase": "OBEY",
                    "image_prompt": "Cyber kittens"
                })}
            
            # Default
            return {"result": "ok"}

        import json
        mock_llm.side_effect = mock_llm_side_effect

        # Run
        result = await manage_bluesky(action="scan_and_reply")
        
        print("\n📝 Result Summary:")
        print(result)

        if "Replied to 1 items" in result:
             print("\n✅ SUCCESS: Creator interaction was replied to!")
        else:
             print("\n❌ FAILURE: Creator interaction was ignored.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_creator_override())
