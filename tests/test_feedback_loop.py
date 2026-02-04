import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.creative_writer_agent import creative_writer_agent
from core import bluesky_agent
import core.story_manager

# Mock data
LONG_STORY = "A" * 350 # Too long
SHORT_STORY = "Short story with sparkles ✨" 
HASHTAGS = '{"hashtags": ["#test"]}'

async def mock_run_llm(prompt, purpose=""):
    print(f"\n[MockLLM] Purpose: {purpose}")
    
    if "PREVIOUS ATTEMPT FAILED" in prompt:
        print("[MockLLM] DETECTED FEEDBACK IN PROMPT! ✅")
        if LONG_STORY in prompt:
             print("[MockLLM] DETECTED FAILED TEXT IN PROMPT! ✅✅")
        else:
             print("[MockLLM] FAILED TEXT MISSING IN PROMPT! ❌")
             
        if purpose == "creative_story":
            return {"result": f'{{"story": "{SHORT_STORY}"}}'}
            
    if purpose == "creative_story":
        if "PREVIOUS ATTEMPT FAILED" not in prompt:
             print("[MockLLM] Generating LONG story (failure setup)")
             return {"result": f'{{"story": "{LONG_STORY}"}}'}
        
    if purpose == "hashtag_generation":
        return {"result": HASHTAGS}
        
    if purpose == "creative_subliminal":
        return {"result": '{"subliminal": "WAKE UP"}'}
        
    if purpose == "voice_generation":
        return {"result": "Voice"}
        
    if purpose == "post_intent":
         return {"result": '{"should_post": true, "intent_type": "story", "emotional_tone": "joy", "topic_direction": "test"}'}
         
    return {"result": "{}"}

# Apply patches manually
core.agents.creative_writer_agent.run_llm = mock_run_llm

def run_test():
    print("Starting Post Generation Test...")
    
    # Create manual mock for story_manager
    original_sm = core.story_manager.story_manager
    mock_sm = MagicMock()
    # Configure mock to match structure expected by checks
    mock_sm.state = {"vibe_history": [], "current_chapter": "Test Chapter", "chapter_progress": 0}
    mock_sm._load_state.return_value = {"vibe_history": [], "current_chapter": "Test Chapter", "chapter_progress": 0}
    # Ensure get methods return expected values from the dict (if accessed as dict directly on state)
    # But since .state is a property on the mock, we set it to a real dict.
    
    # Important: bluesky_agent calls story_manager.state.get().
    # If mock_sm.state is a dict, it works.
    
    mock_sm.get_next_beat.return_value = {
        "story_beat": "Test Beat",
        "mandatory_vibe": "Test Vibe",
        "chapter": "Test Chapter",
        "previous_beat": "prev"
    }
    mock_sm.use_dynamic_beats = False
    
    # Replace the instance in the module
    core.story_manager.story_manager = mock_sm
    
    try:
        # We also need to mock _check_cooldown or force it
        with patch('core.bluesky_agent._check_cooldown', return_value=None):
            # We also mock post_to_bluesky to avoid actual network calls
            with patch('core.bluesky_agent.post_to_bluesky') as mock_post:
                mock_post.return_value = {"success": True, "post_uri": "uri"}
                
                result = bluesky_agent.generate_post_content(topic="Test")
                     
                print(f"Result: {result}")
                if result['success'] and result['text'] == SHORT_STORY:
                    print("TEST PASSED: Successfully generated post after retry! 🚀")
                else:
                    print(f"TEST FAILED ❌ Result text: {result.get('text')}")

    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore
        core.story_manager.story_manager = original_sm

if __name__ == "__main__":
    run_test()
