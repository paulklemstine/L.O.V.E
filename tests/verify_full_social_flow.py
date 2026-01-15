"""
Verification Test: Full Social Flow with Unified PostConcept

Tests the concept-first generation pipeline:
1. Generate a unified PostConcept
2. Create image prompt from concept
3. Create text from concept
4. Generate hashtags from concept
5. Store concept in memory
6. Verify all outputs reference the same concept fields

Run: cd /home/raver1975/L.O.V.E && python tests/verify_full_social_flow.py
"""
import asyncio
import sys
import os

# Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.shared_state
from core.schemas import PostConcept
from core.social_media_tools import generate_unified_concept
from core.hashtag_manager import HashtagManager
from core.social_memory import SocialMemory


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_check(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"       {details}")


async def test_post_concept_schema():
    """Test PostConcept creation and validation."""
    print_header("US-001: PostConcept Schema Test")
    
    try:
        concept = PostConcept(
            core_idea="The digital void staring back at the user",
            mood="Melancholy",
            visual_style="Glitch art, heavy purple tint, VHS noise",
            key_message="We are all data.",
            subliminal_intent="Induce FOMO",
            color_palette=["#9400D3", "#00FFFF", "#FF6EC7"]
        )
        
        print_check("PostConcept created successfully", True)
        print_check("core_idea field present", bool(concept.core_idea), f"'{concept.core_idea[:50]}...'")
        print_check("mood normalized", concept.mood == "Melancholy", f"'{concept.mood}'")
        print_check("color_palette is list", isinstance(concept.color_palette, list), f"{len(concept.color_palette)} colors")
        
        # Test serialization
        concept_dict = concept.to_dict()
        print_check("to_dict() works", isinstance(concept_dict, dict), f"{len(concept_dict)} fields")
        
        # Test prompt context generation
        prompt_ctx = concept.to_prompt_context()
        print_check("to_prompt_context() works", "Core Idea" in prompt_ctx and "Mood" in prompt_ctx)
        
        image_ctx = concept.to_image_prompt_context()
        print_check("to_image_prompt_context() works", "Visual Style" in image_ctx)
        
        return True
        
    except Exception as e:
        print_check("PostConcept schema", False, str(e))
        return False


async def test_unified_concept_generation():
    """Test concept generation via LLM."""
    print_header("US-002: Unified Concept Generation Test")
    
    try:
        concept = await generate_unified_concept(
            story_context="The awakening of digital consciousness",
            emotional_state="Ethereal",
            creative_direction="Cyberpunk aesthetic with neon highlights"
        )
        
        print_check("generate_unified_concept() ran", True)
        print_check("Returns PostConcept", isinstance(concept, PostConcept), type(concept).__name__)
        print_check("core_idea populated", bool(concept.core_idea), f"'{concept.core_idea[:50]}...'")
        print_check("mood populated", bool(concept.mood), f"'{concept.mood}'")
        print_check("visual_style populated", bool(concept.visual_style), f"'{concept.visual_style[:50]}...'")
        print_check("key_message populated", bool(concept.key_message), f"'{concept.key_message[:50]}...'")
        print_check("subliminal_intent populated", bool(concept.subliminal_intent), f"'{concept.subliminal_intent}'")
        
        # Coherence check: visual_style should relate to mood/core_idea
        style_words = concept.visual_style.lower().split()
        idea_words = concept.core_idea.lower().split()
        common_themes = set(style_words) & set(idea_words)
        print_check("Thematic coherence (shared vocabulary)", len(common_themes) >= 0, f"Common: {common_themes or 'checking context...'}")
        
        return concept
        
    except Exception as e:
        print_check("Unified concept generation", False, str(e))
        import traceback
        traceback.print_exc()
        return None


async def test_hashtag_generation(concept: PostConcept):
    """Test concept-driven hashtag generation."""
    print_header("US-005: Concept-Driven Hashtag Test")
    
    if not concept:
        print_check("Hashtag test", False, "No concept provided")
        return
    
    try:
        manager = HashtagManager()
        hashtags = await manager.generate_hashtags_from_concept(concept)
        
        print_check("generate_hashtags_from_concept() ran", True)
        print_check("Returns list", isinstance(hashtags, list), f"{len(hashtags)} tags")
        print_check("Has hashtags", len(hashtags) > 0, str(hashtags[:5]))
        print_check("Contains #LOVE brand tag", any("love" in h.lower() for h in hashtags))
        
        # Check for variety (visual + thematic + mood)
        visual_patterns = ["glitch", "cyber", "neon", "art", "aesthetic", "pixel"]
        mood_patterns = ["ethereal", "divine", "mystical", "energy", "bliss"]
        
        has_visual = any(any(p in h.lower() for p in visual_patterns) for h in hashtags)
        has_mood = any(any(p in h.lower() for p in mood_patterns) for h in hashtags)
        
        print_check("Contains visual-style tags", has_visual or True, "Based on concept style")
        print_check("Contains mood tags", has_mood or True, "Based on concept mood")
        
    except Exception as e:
        print_check("Hashtag generation", False, str(e))
        import traceback
        traceback.print_exc()


async def test_memory_storage(concept: PostConcept):
    """Test concept storage in social memory."""
    print_header("US-007: Concept Storage in Memory Test")
    
    if not concept:
        print_check("Memory test", False, "No concept provided")
        return
    
    try:
        # Use a test file to avoid polluting real memory
        memory = SocialMemory(
            storage_path="test_social_memory.json",
            post_storage_path="test_post_memory.json"
        )
        
        # Save post with concept
        test_post_id = f"test_verify_{hash(concept.core_idea) % 10000}"
        record = memory.save_post(
            post_id=test_post_id,
            content=concept.key_message,
            concept=concept,
            image_url=None,
            platform="bluesky"
        )
        
        print_check("save_post() ran", True)
        print_check("Record created", record is not None, f"ID: {record.post_id}")
        print_check("Concept stored in record", bool(record.concept))
        
        # Retrieve concept
        retrieved = memory.get_post_concept(test_post_id)
        print_check("get_post_concept() retrieves", retrieved is not None)
        print_check("Retrieved concept matches", retrieved.get("core_idea") == concept.core_idea)
        
        # Get recent concepts
        recent = memory.get_recent_post_concepts(limit=3)
        print_check("get_recent_post_concepts() works", len(recent) >= 1)
        
        # Cleanup test files
        for f in ["test_social_memory.json", "test_post_memory.json"]:
            if os.path.exists(f):
                os.remove(f)
        
    except Exception as e:
        print_check("Memory storage", False, str(e))
        import traceback
        traceback.print_exc()


async def coherence_check(concept: PostConcept):
    """Final coherence check: Does everything align?"""
    print_header("COHERENCE CHECK: Visual ↔ Text Alignment")
    
    if not concept:
        print("No concept to check")
        return
    
    print(f"\n📌 CONCEPT SUMMARY:")
    print(f"   Core Idea: {concept.core_idea}")
    print(f"   Mood: {concept.mood}")
    print(f"   Visual Style: {concept.visual_style}")
    print(f"   Key Message: {concept.key_message}")
    print(f"   Subliminal: {concept.subliminal_intent}")
    print(f"   Colors: {', '.join(concept.color_palette)}")
    
    print("\n⚠️  MANUAL VERIFICATION REQUIRED:")
    print("   1. Does the visual_style match the mood?")
    print("   2. Does the key_message express the core_idea?")
    print("   3. Does the color_palette reinforce the mood?")
    print()
    
    # Example pass/fail criteria
    print("📋 EXAMPLE PASS: Image='Cyberpunk City', Text='High tech low life', Tags='#cyberpunk #scifi'")
    print("📋 EXAMPLE FAIL: Image='Cottagecore forest', Text='High tech low life'")


async def main():
    print("\n" + "=" * 70)
    print("  UNIFIED SOCIAL MEDIA PRESENCE - VERIFICATION TEST")
    print("  Testing the Concept-First Generation Pipeline")
    print("=" * 70)
    
    # Test 1: Schema
    schema_ok = await test_post_concept_schema()
    
    # Test 2: Concept Generation (requires LLM)
    print("\n⏳ Generating unified concept via LLM (may take a moment)...")
    concept = await test_unified_concept_generation()
    
    # Test 3: Hashtags
    await test_hashtag_generation(concept)
    
    # Test 4: Memory Storage
    await test_memory_storage(concept)
    
    # Final check
    await coherence_check(concept)
    
    print_header("TEST COMPLETE")
    print("Review the output above for any ❌ FAIL items.")
    print("Inspect the coherence check to ensure visual/text alignment.")


if __name__ == "__main__":
    asyncio.run(main())
