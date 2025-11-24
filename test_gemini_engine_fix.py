import sys
from unittest.mock import MagicMock
import asyncio
import json

session_tool_registry = MagicMock()
session_tool_registry.get_formatted_tool_metadata.return_value = "Session Tools..."

# Mock execute_reasoning_task
async def mock_execute_reasoning_task(*args, **kwargs):
    return {"result": "Raw LLM Response"}

sys.modules['core.llm_api'].execute_reasoning_task = mock_execute_reasoning_task

# Mock smart_parse_llm_response to return a string action
def mock_smart_parse_llm_response(response, expected_keys=None):
    # Simulate the problematic case: action is a string
    return {
        "thought": "I will use a tool.",
        "action": '{"tool_name": "test_tool", "arguments": {}}' 
    }

sys.modules['core.llm_parser'].smart_parse_llm_response = mock_smart_parse_llm_response

async def test():
    engine = GeminiReActEngine(tool_registry)
    # Mock tool registry to return a tool
    mock_tool = MagicMock()
    engine.tool_registry.get_tool.return_value = mock_tool
    
    # Run execute_goal
    # We expect it to parse the string action and call the tool, or at least NOT crash.
    # Since we mocked the tool registry, if it parses correctly, it will try to get the tool.
    
    # We need to mock the tool execution to avoid errors further down
    mock_tool.return_value = "Tool Output"
    
    result = await engine.execute_goal("Test Goal", max_steps=1)
    
    print("Result:", result)
    
    # If it didn't crash, and tried to execute the tool (or handled the error), we are good.
    # In this case, since we provided valid JSON in the string, it should parse it and succeed.
    
    # Let's also test invalid JSON string
    def mock_smart_parse_invalid(response, expected_keys=None):
         return {
            "thought": "I will use a tool.",
            "action": '{"tool_name": "test_tool", "arguments": ... invalid json ...' 
        }
    sys.modules['core.llm_parser'].smart_parse_llm_response = mock_smart_parse_invalid
    
    result_invalid = await engine.execute_goal("Test Goal Invalid", max_steps=1)
    print("Result Invalid:", result_invalid)
    # Should not crash, but log error and continue (max steps reached)
    
    print("Test Passed!")

if __name__ == "__main__":
    asyncio.run(test())
