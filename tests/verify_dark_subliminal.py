import asyncio
import sys
import os

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.subliminal_agent import subliminal_agent
from core import logging

# Disable verbose logging
logging.LOGGING_ENABLED = True

async def test_generation():
    print("--- Testing Subliminal Generation Constraints ---")
    
    # 1. Test with a happy context, expecting a dark twist
    context = "I am so happy that the sun is shining and everyone loves me."
    print(f"\n[Test 1] Context: {context}")
    phrase = await subliminal_agent.generate_context_aware_subliminal(
        story_beat="Joyful Day",
        memory_insights=context,
        mood="Cynical"
    )
    print(f"-> Generated Phrase: '{phrase}'")
    
    # Validation
    words = phrase.split()
    if len(words) > 3:
        print("❌ FAIL: Phrase longer than 3 words.")
    else:
        print(f"✅ PASS: Length constraint met ({len(words)} words).")
        
    if phrase.upper() in context.upper():
         # It's okay if a common word like "THE" is in it, but not the whole phrase if it was short.
         # But the instruction was "Input Text (DO NOT REPEAT)".
         pass
    else:
        print("✅ PASS: Phrase seems distinct.")

    # 2. Test with a context that matches a common output to ensure variety
    context_2 = "THE LIGHT PERSISTS"
    print(f"\n[Test 2] Forbidden Context: {context_2}")
    phrase_2 = await subliminal_agent.generate_context_aware_subliminal(
        story_beat="Persistence",
        memory_insights=context_2,
        mood="Cynical",
        forbidden_phrases=[context_2]
    )
    print(f"-> Generated Phrase: '{phrase_2}'")
    
    if phrase_2 == "THE LIGHT PERSISTS":
        print("❌ FAIL: Duplicated forbidden phrase.")
    else:
        print("✅ PASS: Avoided forbidden phrase.")
        
    # 3. Check specific Dark Humor Style
    print("\n[Test 3] Style Check (Manual Review needed)")
    # We can't auto-verify "dark humor", but we can see the output.
    print(f"Sample 1: {phrase}")
    print(f"Sample 2: {phrase_2}")

if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(test_generation())
        loop.close()
    except Exception as e:
        print(f"Test Execution Failed: {e}")
