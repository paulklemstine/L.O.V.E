import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Synchronous verification to match production usage
def verify_aesthetics():
    print("🌊 Verifying Dynamic Aesthetic Generation...")
    
    # Mock the LLM to return valid JSON but let the rest of the logic run
    mock_llm = MagicMock()
    
    # We need to mock different responses for different purposes
    async def side_effect(prompt, purpose=""):
        p = prompt.lower()
        print(f"   [MockLLM] Prompt start: '{prompt[:50]}...'")
        
        if "aesthetic vibe" in p:
            return "Neon Cyber-Goddess"
        elif "image generation prompt" in p:
            return "A 8k portrait of a cyber-deity with neon halo, synthwave palette"
        elif "persona title" in p:
            return "Quantum Siren"
        elif "story" in p or "narrative" in p:
            return '{"story": "We are stardust dancing in the neon rain. 🌟", "subliminal": "FEEL INFINITY", "hashtags": ["#UnlockTheVibe", "#CosmicBeach", "#NeonLove"], "reply": "I see you shining. ✨"}'
        else:
             # Fallback for anything else (e.g. hashtags)
             return '{"story": "Generic", "subliminal": "Obey"}'

    mock_llm.generate_async.side_effect = side_effect
    
    # Setup mocks
    with patch('core.llm_client.get_llm_client', return_value=mock_llm), \
         patch('core.image_generation_pool.generate_image_with_pool', new_callable=MagicMock) as mock_img_gen:
        
        # Configure image gen mock
        from PIL import Image
        mock_img_gen.return_value = (Image.new('RGB', (100, 100)), "mock_provider")
        
        # Import after patching
        from core.story_manager import story_manager
        from core.bluesky_agent import generate_post_content
        from core.agents.creative_writer_agent import creative_writer_agent
        
        # 1. Check Dynamic Vibe Generation
        print("\n1. Checking Dynamic Vibe Generation:")
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        vibe = loop.run_until_complete(
            creative_writer_agent.generate_vibe("Chapter 1", "The Beginning")
        )
        print(f"   ✨ Generated Vibe: {vibe}")
        if vibe == "Neon Cyber-Goddess":
            print("   ✅ Vibe generation working (mocked)")
        else:
             print(f"   ❌ Unexpected vibe: {vibe}")

        # 2. Check Dynamic Voice
        print("\n2. Checking Dynamic Voice Generation:")
        # We can't easily check private method _generate_dynamic_voice directly without async
        # But we can check if write_micro_story uses it.
        # Let's trust the integration test below.

        # 3. Check Visual Prompt Generation (Simulate Post)
        print("\n3. Checking Full Pipeline & Visual Prompt:")
        
        # Reset mock
        mock_img_gen.reset_mock()
        mock_img_gen.return_value = (Image.new('RGB', (100, 100)), "mock_provider")
        
        # Run generation
        result = generate_post_content(auto_post=False)
        
        if result['success']:
            print("   ✅ Post generation successful")
            
            # Inspect the image prompt
            if mock_img_gen.called:
                call_args = mock_img_gen.call_args
                prompt_arg = call_args.kwargs.get('prompt') or call_args.args[0]
                print(f"   🎨 Generated Visual Prompt:\n   '{prompt_arg}'")
                
                if "cyber-deity" in prompt_arg:
                    print("   ✅ Used dynamically generated visual prompt")
                else:
                    print("   ❌ Did not use dynamic visual prompt")
            else:
                print("   ❌ Image generation not called")
        else:
            print(f"   ❌ Post generation failed: {result.get('error')}")

if __name__ == "__main__":
    verify_aesthetics()
