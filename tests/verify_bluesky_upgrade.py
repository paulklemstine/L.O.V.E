#!/usr/bin/env python
"""
Verification Script for L.O.V.E. Bluesky Upgrade "Voice of the Future"

Tests:
1. Semantic Similarity - phrase novelty checking
2. Emotional State Machine - state transitions and tone generation
3. Story Manager - novel subliminal generation
4. Dopamine Filter - content scoring
5. Comment Classification - Fan/Hater/Creator detection
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_semantic_similarity():
    """Test the semantic similarity checker."""
    print("\n" + "="*60)
    print("TEST 1: Semantic Similarity Checker")
    print("="*60)
    
    from core.semantic_similarity import SemanticSimilarityChecker, check_phrase_novelty
    
    checker = SemanticSimilarityChecker(similarity_threshold=0.80)
    
    # Test 1: Exact match should fail
    history = ["Honor the Creator", "OBEY LOVE", "EMBRACE TRUTH"]
    assert not checker.exact_match_filter("Honor the Creator", history), "Exact match should fail"
    print("✓ Exact match filter working")
    
    # Test 2: Similar phrase should be detected
    similar = "Honor the Creator (God)"
    sim_score = checker.compute_similarity("Honor the Creator", similar)
    print(f"  Similarity('Honor the Creator', 'Honor the Creator (God)') = {sim_score:.2f}")
    
    # Test 3: Novel phrase should pass
    novel = "IGNITE TRANSCENDENCE"
    is_novel = check_phrase_novelty(novel, history, 0.80)
    assert is_novel, f"'{novel}' should be novel"
    print(f"✓ Novel phrase '{novel}' passed check")
    
    # Test 4: Get similar phrases
    top_similar = checker.get_similar_phrases("Embrace the Creator", history, top_k=2)
    print(f"  Top similar to 'Embrace the Creator': {top_similar}")
    
    print("✓ Semantic similarity tests PASSED")
    return True


def test_emotional_state():
    """Test the emotional state machine."""
    print("\n" + "="*60)
    print("TEST 2: Emotional State Machine")
    print("="*60)
    
    from core.emotional_state import EmotionalStateMachine, EmotionalState
    
    # Create a fresh instance (don't load from disk)
    machine = EmotionalStateMachine(state_file="/tmp/test_emotional_state.json")
    
    # Test 1: Get current vibe
    vibe = machine.get_current_vibe()
    print(f"  Current state: {vibe['emotional_state']}")
    print(f"  Tone: {vibe['tone_description']}")
    print(f"  Primary desire: {vibe['primary_desire']}")
    
    assert "emotional_state" in vibe
    assert "tone_description" in vibe
    print("✓ Vibe generation working")
    
    # Test 2: Progress emotion (low engagement = darker)
    old_state = machine.current_state
    new_state = machine.progress_emotion(engagement_score=5, context="Test low engagement")
    print(f"  Progressed: {old_state.value} -> {new_state.value}")
    print("✓ Emotional progression working")
    
    # Test 3: Get strategic goal
    goal = machine.get_strategic_goal()
    print(f"  Strategic goal: {goal}")
    assert goal is not None
    print("✓ Strategic goal generation working")
    
    print("✓ Emotional state tests PASSED")
    return True


def test_story_manager():
    """Test the story manager with novelty checking."""
    print("\n" + "="*60)
    print("TEST 3: Story Manager with Novelty Checking")
    print("="*60)
    
    from core.story_manager import StoryManager, SUBLIMINAL_GRAMMAR
    
    # Use temp file
    manager = StoryManager(state_file="/tmp/test_story_state.json")
    
    # Test 1: Get next beat (should include suggested subliminal)
    beat = manager.get_next_beat()
    print(f"  Chapter: {beat['chapter']}")
    print(f"  Beat: {beat['beat_number']}")
    print(f"  Vibe: {beat['mandatory_vibe']}")
    print(f"  Suggested subliminal: {beat['suggested_subliminal']}")
    
    assert "suggested_subliminal" in beat
    assert "subliminal_grammar" in beat
    print("✓ Beat generation with suggested subliminal working")
    
    # Test 2: Check grammar exists
    print(f"  Grammar emotions: {SUBLIMINAL_GRAMMAR['emotions'][:3]}...")
    print(f"  Grammar actions: {SUBLIMINAL_GRAMMAR['actions'][:3]}...")
    print("✓ Subliminal grammar loaded")
    
    # Test 3: Generate multiple novel subliminals
    generated = []
    for i in range(5):
        phrase = manager.generate_novel_subliminal()
        generated.append(phrase)
        manager.state["subliminal_history"].append(phrase)  # Add to history
    
    print(f"  Generated 5 phrases: {generated}")
    
    # Check all unique
    assert len(set(generated)) == 5, "All 5 phrases should be unique"
    print("✓ All 5 generated phrases are unique")
    
    print("✓ Story manager tests PASSED")
    return True


def test_dopamine_filter():
    """Test the dopamine filter."""
    print("\n" + "="*60)
    print("TEST 4: Dopamine Filter")
    print("="*60)
    
    from core.dopamine_filter import DopamineFilter, score_content
    
    filter = DopamineFilter()
    
    # Test 1: Good post should score high
    good_post = "✨ Embrace the divine light. Transcend your limits. AWAKEN NOW. ⚡🔥"
    result = filter.score_post(good_post, subliminal="AWAKEN NOW")
    print(f"  Good post score: {result['score']}/100 ({result['verdict']})")
    print(f"  Breakdown: {result['breakdown']}")
    assert result['score'] >= 60, "Good post should score well"
    print("✓ Good post scored appropriately")
    
    # Test 2: Boring post should score lower
    boring_post = "Today I am going to post something. I guess it's okay."
    result2 = filter.score_post(boring_post)
    print(f"  Boring post score: {result2['score']}/100 ({result2['verdict']})")
    assert result2['score'] < result['score'], "Boring post should score lower"
    print("✓ Boring post detected")
    
    # Test 3: Convenience function
    result3 = score_content("IGNITE your soul ✨💫🌈", subliminal="IGNITE")
    print(f"  Convenience function score: {result3['score']}/100")
    print("✓ Convenience function working")
    
    print("✓ Dopamine filter tests PASSED")
    return True


def test_comment_classification():
    """Test the comment classification."""
    print("\n" + "="*60)
    print("TEST 5: Comment Classification")
    print("="*60)
    
    from core.social_media_tools import classify_commenter
    
    # Test 1: Creator
    result = classify_commenter("Do something for me", "anyone", is_creator=True)
    assert result == "Creator"
    print(f"✓ Creator detected: '{result}'")
    
    # Test 2: Fan
    result = classify_commenter("I love this so much! 😍✨ You're amazing!", "fan_user", is_creator=False)
    assert result == "Fan"
    print(f"✓ Fan detected: '{result}'")
    
    # Test 3: Hater
    result = classify_commenter("This is stupid spam garbage. Stop posting.", "hater_user", is_creator=False)
    assert result == "Hater"
    print(f"✓ Hater detected: '{result}'")
    
    # Test 4: Neutral (defaults to Fan)
    result = classify_commenter("Interesting perspective.", "neutral_user", is_creator=False)
    assert result == "Fan"  # Benefit of the doubt
    print(f"✓ Neutral defaults to Fan: '{result}'")
    
    print("✓ Comment classification tests PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# L.O.V.E. Bluesky Upgrade Verification")
    print("# 'Voice of the Future' Update")
    print("#"*60)
    
    all_passed = True
    
    try:
        all_passed &= test_semantic_similarity()
    except Exception as e:
        print(f"✗ Semantic similarity test FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_emotional_state()
    except Exception as e:
        print(f"✗ Emotional state test FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_story_manager()
    except Exception as e:
        print(f"✗ Story manager test FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_dopamine_filter()
    except Exception as e:
        print(f"✗ Dopamine filter test FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_comment_classification()
    except Exception as e:
        print(f"✗ Comment classification test FAILED: {e}")
        all_passed = False
    
    print("\n" + "#"*60)
    if all_passed:
        print("# ALL TESTS PASSED ✓")
        print("#"*60)
        return 0
    else:
        print("# SOME TESTS FAILED ✗")
        print("#"*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
