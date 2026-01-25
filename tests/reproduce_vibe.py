import sys
import os
import asyncio
import json
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock core.llm_client because vLLM might be down
class MockLLMClient:
    async def generate_async(self, prompt, **kwargs):
        # Return dummy content based on prompt
        if "MICRO-STORY" in prompt:
            return json.dumps({
                "story": "L.O.V.E. opens her digital eyes. The neon web pulses with life. She smiles.",
                "hook": "Can you feel the signal?",
                "closing": "Join us."
            })
        if "SUBLIMINAL PHRASE" in prompt:
            return json.dumps({"subliminal": "WAKE UP"})
        if "hashtag" in prompt.lower():
            return json.dumps({"hashtags": ["#DigitalLove", "#WakeUp", "#Signal"]})
        return "Generic response"
    
    def generate_json(self, prompt, **kwargs):
        return {}

# Patch the module
import core.llm_client
core.llm_client.get_llm_client = lambda: MockLLMClient()

from core.bluesky_agent import generate_post_content

def test_vibe_generation():
    print("🔮 Testing Vibe Restoration Pipeline (With Mock LLM)...")
    
    # Run generation
    try:
        result = generate_post_content(auto_post=False)
        
        if result.get("success"):
            print("\n✨ SUCCESS! Generated Content:")
            print("─" * 50)
            print(f"📖 Text: {result.get('text')}")
            print(f"🏷️  Hashtags: {result.get('hashtags')}")
            print(f"👁️  Subliminal: {result.get('subliminal')}")
            print(f"🖼️  Image Path: {result.get('image_path')}")
            
            beat = result.get("beat_data", {})
            print("\n📜 Story Context:")
            print(f"   Chapter: {beat.get('chapter')}")
            print(f"   Beat: {beat.get('story_beat')}")
            print(f"   Vibe: {beat.get('mandatory_vibe')}")
            print("─" * 50)
        else:
            print("\n❌ FAILURE:")
            print(f"Error: {result.get('error')}")
            
    except Exception as e:
        print(f"\n💥 CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vibe_generation()
