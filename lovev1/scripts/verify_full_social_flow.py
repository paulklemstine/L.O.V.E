import asyncio
import os
import sys
import logging
import json
import time
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

from core.nodes.social_media_team import _create_and_post_story_segment, _load_story_state, _handle_interactions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    load_dotenv()
    print("--- Starting Social Media Flow Verification ---")
    
    # 1. Load State
    print("\n[1] Loading Story State...")
    state = _load_story_state()
    full_state_mock = {"story_arc": state.get("story_arc", {}), "processed_cids": state.get("processed_cids", {})}
    print(f"Current Chapter: {full_state_mock['story_arc'].get('current_chapter')}")
    
    # 2. Test Content Generation & Posting
    print("\n[2] Testing Content Generation & Posting (Live)...")
    try:
        # Retry logic for the main call
        result = None
        for i in range(3):
            try:
                print(f"Attempt {i+1}...")
                result = await _create_and_post_story_segment(full_state_mock)
                break
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if "429" in str(e):
                    print("Rate limit hit. Sleeping 10s...")
                    await asyncio.sleep(10)
                else:
                    import traceback
                    traceback.print_exc()
                    break
        
        if result:
            print(f"SUCCESS Result: {result}")
        else:
             print("FAILURE: Could not post after retries.")

    except Exception as e:
        print(f"CRITICAL ERROR in posting: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test Interaction/Comment Check
    print("\n[3] Testing Interaction Check...")
    try:
        interaction_result = await _handle_interactions()
        print(f"Interaction Result: {interaction_result}")
    except Exception as e:
        print(f"ERROR in interactions: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
