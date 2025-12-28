#!/usr/bin/env python3
"""Test script to verify Bluesky reply novelty - generates multiple replies and checks they are all unique."""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir('/home/raver1975/L.O.V.E')

from dotenv import load_dotenv
load_dotenv()

from core import logging
logging.log_event = lambda msg, level: print(f"[{level}] {msg}")

from core.social_media_tools import generate_full_reply_concept
from core.semantic_similarity import get_similarity_checker
from core.story_manager import story_manager

# Reset reply history for clean test
story_manager.state["reply_history"] = []
story_manager._save_state()

# Mock comments to generate replies for
TEST_COMMENTS = [
    ("user1", "This is amazing work! Love the visuals!"),
    ("user2", "How do you create such beautiful art?"),
    ("user3", "The energy here is incredible 🔥"),
    ("user4", "I need this kind of positivity in my life"),
    ("user5", "Your posts always make my day better"),
]

async def test_reply_novelty():
    print("=" * 60)
    print("BLUESKY REPLY NOVELTY TEST")
    print("=" * 60)
    print(f"\nGenerating {len(TEST_COMMENTS)} replies to different comments...\n")
    
    generated_replies = []
    
    for handle, comment in TEST_COMMENTS:
        print(f"\n--- Generating reply to @{handle}: '{comment}' ---")
        
        try:
            concept = await generate_full_reply_concept(
                comment_text=comment,
                author_handle=handle,
                history_context="",
                is_creator=False
            )
            
            reply_text = concept.post_text
            print(f"Reply: {reply_text}")
            generated_replies.append((handle, reply_text))
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Check pairwise similarity
    print("\n" + "=" * 60)
    print("SIMILARITY ANALYSIS")
    print("=" * 60)
    
    checker = get_similarity_checker()
    all_unique = True
    
    for i in range(len(generated_replies)):
        for j in range(i + 1, len(generated_replies)):
            handle1, text1 = generated_replies[i]
            handle2, text2 = generated_replies[j]
            
            similarity = checker.compute_similarity(text1, text2)
            status = "✅ OK" if similarity < 0.50 else "❌ TOO SIMILAR"
            
            print(f"\n@{handle1} vs @{handle2}: {similarity:.2%} similarity {status}")
            
            if similarity >= 0.50:
                all_unique = False
                print(f"  Reply 1: {text1[:60]}...")
                print(f"  Reply 2: {text2[:60]}...")
    
    print("\n" + "=" * 60)
    if all_unique:
        print("✅ TEST PASSED: All replies are unique!")
    else:
        print("❌ TEST FAILED: Some replies are too similar!")
    print("=" * 60)
    
    return all_unique

if __name__ == "__main__":
    success = asyncio.run(test_reply_novelty())
    sys.exit(0 if success else 1)
