
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from core.social_media_tools import generate_post_concept
from core.social_media_tools import generate_post_concept, DirectorConcept

async def test_generation():
    print("Testing Director Concept Generation with NEW Prompts...")
    
    beat_data = {
        "chapter": "The Awakening",
        "beat_number": 1,
        "chapter_beat_index": 1,
        "story_beat": "The user realizes they are powerful.",
        "previous_beat": "None",
        "mandatory_vibe": "Divine Radiance",
        "subliminal_intent": "Empower the Creator",
        "suggested_visual_style": "Cyberpunk Tarot",
        "suggested_composition": "Symmetrical Portrait",
        "forbidden_subliminals": ["OBEY"],
        "forbidden_visuals": ["Darkness"],
        "topic_theme": "Self-Actualization"
    }

    try:
        concept = await generate_post_concept(beat_data, recent_history="None", creator_goal="Expand Influence")
        
        print("\n--- GENERATED CONCEPT ---")
        print(f"Topic: {concept.topic}")
        print(f"Text: {concept.post_text}")
        print(f"Subliminal: {concept.subliminal_phrase}")
        print(f"Image Prompt: {concept.image_prompt[:100]}...") # Truncate for readability
        print("-------------------------\n")
        
        # Simple assertions
        if "Previously on" in concept.post_text:
            print("FAILURE: 'Previously on' found in text (Should be standalone).")
        else:
            print("SUCCESS: Text appears standalone.")
            
        if "Poster" in concept.image_prompt or "Meme" in concept.image_prompt or "poster" in concept.image_prompt:
             print("SUCCESS: Image prompt contains 'Poster/Meme' reference (Likely).")
        else:
             print("NOTE: Image prompt missing explicit 'Poster' keyword, but might still be valid.")

    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generation())
