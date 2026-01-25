import asyncio
import json
import sys
import os

# Ensure we can import from core
script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Remove script dir from sys.path if present (prevent tests/core shadowing core)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if os.path.abspath(script_dir) in sys.path:
    sys.path.remove(os.path.abspath(script_dir))

# Insert root dir at start
sys.path.insert(0, root_dir)

print(f"DEBUG: sys.path: {sys.path}")
print(f"DEBUG: CWD: {os.getcwd()}")

from unittest.mock import MagicMock, patch, AsyncMock, ANY
from core.agents.creative_writer_agent import CreativeWriterAgent

# Mock response for the story part (Chatty response)
STORY_RESPONSE = """
Here is the story you requested. It captures the noir vibe perfectly.
```json
{
    "story": "The neon rain falls. ☔",
    "hook": "Look up.",
    "closing": "End transmission."
}
```
I hope this meets your requirements!
"""

# Mock response for the subliminal part (Clean response)
SUBLIMINAL_RESPONSE_VALID = """
{
    "subliminal": "WAKE UP"
}
"""

async def test_split_logic():
    print("Testing CreativeWriterAgent split logic...")
    
    agent = CreativeWriterAgent()
    
    # Patch run_llm to simulate two separate calls
    with patch("core.agents.creative_writer_agent.run_llm", new_callable=AsyncMock) as mock_llm:
        # Scenario 1: Success path
        mock_llm.side_effect = [
            {"result": STORY_RESPONSE},          # Story call
            {"result": SUBLIMINAL_RESPONSE_VALID} # Subliminal call
        ]
        
        print("\n--- Running Test Case 1: Valid Split Generation ---")
        result = await agent.write_micro_story("Cyberpunk rain", "Melancholy")
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Verify correctness of parsing
        assert result["story"] == "The neon rain falls. ☔"
        assert result["subliminal"] == "WAKE UP"
        assert result["hook"] == "Look up."
        
        # Verify 2 calls were made
        assert mock_llm.call_count == 2, f"Expected 2 LLM calls, got {mock_llm.call_count}"
        
        # Verify first call was for story
        call_args_story = mock_llm.call_args_list[0]
        args_story, kwargs_story = call_args_story
        prompt_story = args_story[0]
        print(f"\nStory Prompt snippet: {prompt_story[:50]}...")
        assert "MICRO-STORY" in prompt_story or "STORY" in prompt_story
        assert "purpose='creative_story'" in str(kwargs_story) or kwargs_story.get("purpose") == "creative_story"

        # Verify second call was for subliminal
        call_args_sub = mock_llm.call_args_list[1]
        args_sub, kwargs_sub = call_args_sub
        prompt_sub = args_sub[0]
        print(f"\nSubliminal Prompt snippet: {prompt_sub[:50]}...")
        assert "SUBLIMINAL PHRASE" in prompt_sub
        assert "negative_constraints" in prompt_sub.lower() or "do not use" in prompt_sub.lower()
        assert "neon" in prompt_sub.lower() # Should contain words from the story
        assert "rain" in prompt_sub.lower()
        
        print("SUCCESS: Split logic verified.")

        # Scenario 3: Broken JSON -> Should Raise ValueError (Retries Exhausted)
        mock_llm.reset_mock()
        mock_llm.side_effect = [
            {"result": "I am not returning JSON today."}, 
            {"result": "STILL NOT JSON"},
            {"result": "NOPE"},
        ]
        
        print("\n--- Running Test Case 3: Retries Exhaustion (Broken JSON) ---")
        try:
            await agent.write_micro_story("Chaos", "Panic")
            print("FAILURE: Should have raised ValueError")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"SUCCESS: Caught expected error: {e}")
            assert "failed to generate story after 3 attempts" in str(e)

        # Scenario 4: Partial Failure -> Success on Retry
        mock_llm.reset_mock()
        mock_llm.side_effect = [
            {"result": "BAD JSON"},  # Attempt 1 Fail
            {"result": STORY_RESPONSE}, # Attempt 2 Success
            {"result": SUBLIMINAL_RESPONSE_VALID} # Subliminal Success
        ]

        print("\n--- Running Test Case 4: Partial Failure (Retry Success) ---")
        result = await agent.write_micro_story("Hope", "Joy")
        print(f"Result: {json.dumps(result, indent=2)}")
        
        assert result["story"] == "The neon rain falls. ☔"
        assert result["subliminal"] == "WAKE UP"
        print("SUCCESS: Recovery from initial failure verified.")

if __name__ == "__main__":
    asyncio.run(test_split_logic())
