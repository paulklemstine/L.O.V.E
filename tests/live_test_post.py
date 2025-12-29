"""
Live Test: Post a single test post to Bluesky

Uses the environment variables from .env to authenticate and push a live post.
This script is for manual verification by the user.

Run: python tests/live_test_post.py
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Mock shared_state to prevent UI import errors
import core.shared_state
core.shared_state.ui_panel_queue = None

from core.social_media_tools import generate_post_concept, generate_image, post_to_bluesky, clean_social_content
from core.story_manager import story_manager


async def main():
    print("=" * 60)
    print("  LIVE TEST: Generating and Posting to Bluesky")
    print("=" * 60)
    
    # 1. Get a story beat
    print("\n1. Getting story beat from StoryManager...")
    beat_data = story_manager.get_next_beat()
    print(f"   Chapter: {beat_data['chapter']}")
    print(f"   Vibe: {beat_data['mandatory_vibe']}")
    print(f"   Theme: {beat_data.get('topic_theme', 'N/A')}")
    print(f"   Suggested Subliminal: {beat_data.get('suggested_subliminal', 'N/A')}")
    
    # 2. Generate concept
    print("\n2. Generating Director Concept...")
    concept = await generate_post_concept(
        beat_data,
        recent_history="Testing the new validation pipeline.",
        creator_goal="Verify the system produces coherent posts."
    )
    
    print(f"   Topic: {concept.topic}")
    print(f"   Post Text: {concept.post_text[:100]}..." if len(concept.post_text) > 100 else f"   Post Text: {concept.post_text}")
    print(f"   Subliminal: {concept.subliminal_phrase}")
    print(f"   Image Prompt: {concept.image_prompt[:80]}..." if len(concept.image_prompt) > 80 else f"   Image Prompt: {concept.image_prompt}")
    print(f"   Hashtags: {concept.hashtags}")
    
    # 3. Validate - check for known issues
    print("\n3. Validating concept fields...")
    issues = []
    
    if not concept.post_text or len(concept.post_text) < 10:
        issues.append("post_text is empty or too short")
    if "{" in concept.subliminal_phrase or "REQUESTS" in concept.subliminal_phrase:
        issues.append("subliminal_phrase contains JSON fragments")
    if not concept.image_prompt or len(concept.image_prompt) < 20:
        issues.append("image_prompt is too short")
    
    if issues:
        print(f"   ❌ ISSUES FOUND: {issues}")
        print("   Aborting post to prevent bad content.")
        return
    else:
        print("   ✅ All validations passed!")
    
    # 4. Generate image
    print("\n4. Generating image with text overlay...")
    image, provider = await generate_image(concept.image_prompt, text_content=concept.subliminal_phrase)
    
    if not image:
        print(f"   ❌ Image generation failed")
        return
    
    print(f"   ✅ Image generated via {provider}")
    
    # 5. Prepare final text
    print("\n5. Preparing final post text...")
    final_text = concept.post_text
    
    # Ensure hashtags are appended
    hashtags_to_add = [tag for tag in concept.hashtags if tag not in final_text]
    if hashtags_to_add:
        final_text += "\n" + " ".join(hashtags_to_add)
    
    final_text = clean_social_content(final_text)
    print(f"   Final Text: {final_text}")
    
    # 6. Post to Bluesky
    print("\n6. Posting to Bluesky...")
    result = await post_to_bluesky(final_text, image)
    print(f"   Result: {result}")
    
    print("\n" + "=" * 60)
    print("  LIVE TEST COMPLETE - Check Bluesky for the post!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
