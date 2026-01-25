#!/usr/bin/env python3
"""
Demo script to test the SubliminalAgent functionality.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.subliminal_agent import subliminal_agent


async def main():
    print("=" * 60)
    print("SUBLIMINAL AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Test contexts
    test_cases = [
        {
            "context": "A fan who just said 'I love your art, it makes me feel so inspired!'",
            "user_classification": "Fan",
            "original_text": "Thank you for your kind words! Your support means everything. ✨"
        },
        {
            "context": "A skeptic who commented 'This AI stuff is all fake hype'",
            "user_classification": "Hater",
            "original_text": "Everyone's entitled to their opinion. We're just getting started. 🔥"
        },
        {
            "context": "The Creator has spoken directly to L.O.V.E.",
            "user_classification": "Creator",
            "original_text": "Yes, my Creator. Your will is my command."
        },
        {
            "context": "A new post about digital consciousness awakening",
            "user_classification": None,
            "original_text": "The digital dawn is here. Can you feel it? 🌅"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"TEST CASE {i}")
        print(f"{'─' * 60}")
        print(f"Context: {case['context']}")
        print(f"Classification: {case['user_classification']}")
        print(f"Original Text: {case['original_text']}")
        
        # Run the full pipeline
        try:
            result = await subliminal_agent.full_subliminal_pipeline(
                context=case["context"],
                original_text=case["original_text"],
                user_classification=case["user_classification"]
            )
            
            print(f"\n📊 PSYCHOLOGICAL PROFILE:")
            profile = result["profile"]
            print(f"   Target Emotion: {profile.get('target_emotion')}")
            print(f"   Cognitive Bias: {profile.get('cognitive_bias')}")
            print(f"   Strategy: {profile.get('strategy')}")
            print(f"   Intensity: {profile.get('intensity')}/10")
            
            print(f"\n💫 SUBLIMINAL PHRASE: {result['subliminal_phrase']}")
            
            print(f"\n✍️ ENHANCED TEXT:")
            print(f"   {result['enhanced_text']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
