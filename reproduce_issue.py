import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.deep_agent_engine import DeepAgentEngine
from core.tools import ToolRegistry

async def reproduce():
    # Setup
    registry = ToolRegistry()
    registry.register_tool("test_tool", lambda x: f"result: {x}", {"description": "test", "arguments": {}})

    engine = DeepAgentEngine(api_url="http://mock", tool_registry=registry)

    # We want to trigger the code path where tool_name is "test_tool"
    # Since run() makes a network call, we need to patch httpx.AsyncClient

    mock_response = {
        "choices": [{
            "text": json.dumps({
                "thought": "I will use test_tool",
                "action": {
                    "tool_name": "test_tool",
                    "arguments": {"x": "hello"}
                }
            })
        }]
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.__aenter__.return_value = mock_client_instance

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = mock_response

        # Use AsyncMock for the post method
        mock_client_instance.post = AsyncMock(return_value=mock_post_response)

        print("Attempting to run engine...")
        try:
            result = await engine.run("some prompt")
            print(f"Result: {result}")
        except AttributeError as e:
            print(f"Caught expected error: {e}")
        except Exception as e:
            print(f"Caught unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce())
