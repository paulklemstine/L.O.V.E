
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.social_media_tools import generate_post_concept
from core.prompt_manager import PromptManager
import core.logging

# Setup basic logging to see output
# We might need to mock or initialize shared_state if logging depends on it,
# but verify_social_prompt.py usually runs standalone if imports allow.
import core.shared_state
core.shared_state.ui_panel_queue = None # Mock if needed

async def test_generation():
    print("--- Starting Social Prompt Verification ---")
    
    # Mock Data
    beat_data = {
        "chapter": "The Awakening",
        "beat_number": 42,
        "mandatory_vibe": "Neon Glitch",
        "forbidden_subliminals": ["OBEY", "CONSUME"],
        "forbidden_visuals": ["Red", "Dark"]
    }
    
    recent_history = "Posted about digital rain. Posted about electric sheep."
    creator_goal = "Ensure L.O.V.E. appears as a benevolent digital deity."
    
    print(f"Testing with Creator Goal: {creator_goal}")
    print(f"Testing with History: {recent_history}")
    
    try:
        concept = await generate_post_concept(beat_data, recent_history=recent_history, creator_goal=creator_goal)
        print("\n--- GENERATED CONCEPT ---")
        print(f"Topic: {concept.topic}")
        print(f"Text: {concept.post_text}")
        print(f"Subliminal: {concept.subliminal_phrase}")
        print(f"Image Prompt: {concept.image_prompt}")
        print("\n--- SUCCESS ---")
    except Exception as e:
        print(f"\n--- FAILED: {e} ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generation())
