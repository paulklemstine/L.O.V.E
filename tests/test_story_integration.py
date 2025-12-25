import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.story_manager import story_manager
from core.social_media_tools import generate_post_concept, generate_full_reply_concept
import core.logging

# Setup basic logging to stdout
core.logging.log_event = print

async def test_story_flow():
    print("\n--- Testing Story Manager Beat Generation ---")
    beat_data = story_manager.get_next_beat()
    print(f"Beat Data: {beat_data}")
    
    print("\n--- Testing Director Post Concept Generation ---")
    # This will trigger an LLM call. Ensure environment is set up.
    try:
        concept = await generate_post_concept(beat_data)
        print(f"Generated Concept: {concept}")
        
        if concept.topic == "Fallback":
            print("FAILURE: Fell back to default.")
        else:
            print("SUCCESS: Concept generated via LLM.")
            
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing Director Reply Generation ---")
    try:
        reply = await generate_full_reply_concept("I love the new vibe!", "test_user", "history...", is_creator=False)
        print(f"Reply Concept: {reply}")
        
        creator_reply = await generate_full_reply_concept("New Task: Expand.", "evildrgemini.bsky.social", "history...", is_creator=True)
        print(f"Creator Reply Concept: {creator_reply}")
        
    except Exception as e:
        print(f"ERROR Reply: {e}")

if __name__ == "__main__":
    asyncio.run(test_story_flow())
