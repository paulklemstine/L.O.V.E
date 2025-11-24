import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.deep_agent_engine import DeepAgentEngine
from core.tools import ToolRegistry

async def test_repair():
    print("Starting JSON repair test...")
    registry = ToolRegistry()
    registry.register_tool("test_tool", lambda x: f"result: {x}", {"description": "test", "arguments": {}})

    engine = DeepAgentEngine(api_url="http://mock", tool_registry=registry)

    # Malformed response (legacy keys)
    malformed_response_text = json.dumps({
        "type": "I will use test_tool",
        "goal": {
            "tool_name": "test_tool",
            "arguments": {"x": "hello"}
        }
    })
    
    # Corrected response (what the repair LLM should return)
    repaired_response_text = json.dumps({
        "thought": "I will use test_tool",
        "action": {
            "tool_name": "test_tool",
            "arguments": {"x": "hello"}
        }
    })

    # Mocking the network calls
    # First call returns malformed, second call (repair) returns corrected
    
    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.__aenter__.return_value = mock_client_instance

        # We need to simulate two different responses for the two calls to generate/post
        # 1. The initial generation -> returns malformed
        # 2. The repair generation -> returns corrected
        
        mock_post_response_1 = MagicMock()
        mock_post_response_1.status_code = 200
        mock_post_response_1.json.return_value = {"choices": [{"text": malformed_response_text}]}
        
        mock_post_response_2 = MagicMock()
        mock_post_response_2.status_code = 200
        mock_post_response_2.json.return_value = {"choices": [{"text": repaired_response_text}]}

        # Side effect for post: first call returns malformed, second returns repaired
        mock_client_instance.post = AsyncMock(side_effect=[mock_post_response_1, mock_post_response_2])

        print("Attempting to run engine with malformed input...")
        result = await engine.run("some prompt")
        print(f"Final Result: {result}")
        
        if "Tool test_tool executed" in str(result):
            print("SUCCESS: JSON was repaired and tool was executed.")
        else:
            print("FAILURE: JSON repair did not work as expected.")

if __name__ == "__main__":
    asyncio.run(test_repair())
