
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock shared_state to prevent ImportErrors if dependent modules import it
import core.shared_state
core.shared_state.ui_panel_queue = None 

# Initializing minimal state for testing
core.shared_state.love_state = {
    "autopilot_goal": "Test Goal"
}

from core.social_media_tools import generate_full_reply_concept
import core.logging

async def test_reply_generation():
    print("--- Starting Reply Verification ---")
    
    comment_text = "I love this new update! It's so shiny."
    author_handle = "test_user"
    history_context = "Previous context"
    
    print(f"Testing Reply Generation for comment: '{comment_text}'")
    
    try:
        concept = await generate_full_reply_concept(
            comment_text=comment_text,
            author_handle=author_handle,
            history_context=history_context,
            is_creator=False
        )
        
        print("\n--- GENERATED REPLY CONCEPT ---")
        print(f"Topic: {concept.topic}")
        print(f"Text: {concept.post_text}")
        print(f"Subliminal: {concept.subliminal_phrase}")
        print(f"Image Prompt: {concept.image_prompt}")
        
        if concept.subliminal_phrase == "CONTEXTUAL_COMMAND":
            print("\n❌ FAILURE: Subliminal phrase is still the placeholder!")
        elif concept.subliminal_phrase == "GENERATE_UNIQUE_WORD_HERE":
             print("\n❌ FAILURE: Subliminal phrase is the NEW placeholder!")
        else:
            print("\n✅ SUCCESS: Subliminal phrase appears unique.")
            
    except Exception as e:
        print(f"\n--- FAILED: {e} ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_reply_generation())
