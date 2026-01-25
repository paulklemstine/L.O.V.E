"""
Test suite for the Final Draft Fixer agent
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.final_draft_fixer import (
    fix_final_draft,
    detect_metadata_leakage,
    detect_duplicate_hashtags,
    auto_fix_metadata_leakage,
    auto_fix_duplicate_hashtags
)


def test_metadata_leakage_detection():
    """Test that we detect metadata leakage patterns"""
    print("Testing metadata leakage detection...")
    
    # Test case 1: (Max 280 chars)
    text1 = "You exist to shine! Your code is a spark for infinite love. 🌟✨ #Spark #GlitchCore #Glitter #Blessing (Max 280 chars)"
    issues1 = detect_metadata_leakage(text1)
    assert len(issues1) > 0, "Should detect '(Max 280 chars)'"
    print(f"✓ Detected: {issues1[0].details}")
    
    # Test case 2: Caption: prefix
    text2 = "Caption: This is awesome! ✨ #LOVE"
    issues2 = detect_metadata_leakage(text2)
    assert len(issues2) > 0, "Should detect 'Caption:' prefix"
    print(f"✓ Detected: {issues2[0].details}")
    
    # Test case 3: Clean text
    text3 = "This is clean! ✨ #LOVE #Divine"
    issues3 = detect_metadata_leakage(text3)
    assert len(issues3) == 0, "Should NOT detect issues in clean text"
    print("✓ Clean text passed")
    
    print()


def test_duplicate_hashtag_detection():
    """Test that we detect duplicate hashtags"""
    print("Testing duplicate hashtag detection...")
    
    # Test case: Duplicate #GlitchCore
    text1 = "Amazing vibes! #LOVE #GlitchCore #Spark #GlitchCore"
    issues1 = detect_duplicate_hashtags(text1)
    assert len(issues1) > 0, "Should detect duplicate #GlitchCore"
    assert "#glitchcore" in issues1[0].details.lower(), "Should report the duplicate tag"
    print(f"✓ Detected: {issues1[0].details}")
    
    # Test case: No duplicates
    text2 = "Amazing vibes! #LOVE #GlitchCore #Spark"
    issues2 = detect_duplicate_hashtags(text2)
    assert len(issues2) == 0, "Should NOT detect duplicates"
    print("✓ Unique hashtags passed")
    
    print()


def test_auto_fix_metadata():
    """Test that auto-fix removes metadata patterns"""
    print("Testing auto-fix for metadata leakage...")
    
    # Test the problematic example from the user
    original = "You exist to shine! Your code is a spark for infinite love. 🌟✨ #Spark #GlitchCore #Glitter #Blessing (Max 280 chars)"
    fixed = auto_fix_metadata_leakage(original)
    
    assert "(Max 280 chars)" not in fixed, "Should remove (Max 280 chars)"
    assert "You exist to shine!" in fixed, "Should preserve the actual content"
    print(f"Original: {original}")
    print(f"Fixed:    {fixed}")
    print("✓ Metadata removed successfully")
    
    print()


def test_auto_fix_duplicates():
    """Test that auto-fix removes duplicate hashtags"""
    print("Testing auto-fix for duplicate hashtags...")
    
    original = "Amazing energy! #LOVE #GlitchCore #Spark #GlitchCore #LOVE"
    fixed = auto_fix_duplicate_hashtags(original)
    
    # Count occurrences
    glitchcore_count = fixed.lower().count("#glitchcore")
    love_count = fixed.lower().count("#love")
    
    assert glitchcore_count == 1, f"Should have only 1 #GlitchCore, found {glitchcore_count}"
    assert love_count == 1, f"Should have only 1 #LOVE, found {love_count}"
    print(f"Original: {original}")
    print(f"Fixed:    {fixed}")
    print("✓ Duplicates removed successfully")
    
    print()


async def test_full_fix_integration():
    """Test the full fix_final_draft function"""
    print("Testing full fix_final_draft integration...")
    
    # The problematic post from the user
    problematic_post = """You exist to shine! Your code is a spark for infinite love. 🌟✨ #Spark #GlitchCore #Glitter #Blessing (Max 280 chars).
#Motivation #GlitchCore"""
    
    print(f"Original post ({len(problematic_post)} chars):")
    print(problematic_post)
    print()
    
    result = await fix_final_draft(problematic_post, auto_fix_only=True)
    
    print(f"Fixed post ({len(result['fixed_text'])} chars):")
    print(result['fixed_text'])
    print()
    
    print(f"Issues found: {len(result['issues'])}")
    for issue in result['issues']:
        print(f"  - {issue}")
    print()
    
    # Verify fixes
    assert "(Max 280 chars)" not in result['fixed_text'], "Should remove metadata"
    assert result['fixed_text'].lower().count("#glitchcore") == 1, "Should remove duplicate #GlitchCore"
    assert result['was_modified'], "Should report modifications"
    
    print("✓ Full integration test passed!")
    print()


def main():
    """Run all tests"""
    print("=" * 70)
    print("FINAL DRAFT FIXER TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        # Sync tests
        test_metadata_leakage_detection()
        test_duplicate_hashtag_detection()
        test_auto_fix_metadata()
        test_auto_fix_duplicates()
        
        # Async test
        asyncio.run(test_full_fix_integration())
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
