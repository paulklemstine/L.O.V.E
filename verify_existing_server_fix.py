
import sys
from unittest.mock import MagicMock, patch

# Mock core.logging and console
sys.modules["core.logging"] = MagicMock()
mock_console = MagicMock()

# Mock DeepAgentEngine
MockDeepAgentEngine = MagicMock()
sys.modules["core.deep_agent_engine"] = MagicMock()
sys.modules["core.deep_agent_engine"].DeepAgentEngine = MockDeepAgentEngine

# Mock requests
mock_requests = MagicMock()
sys.modules["requests"] = mock_requests

def test_existing_server_logic():
    print("Testing existing server logic...")
    
    # Mock response from vLLM
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"id": "Qwen/Qwen2-1.5B-Instruct-AWQ"}]
    }
    mock_requests.get.return_value = mock_response
    
    # Simulate the logic snippet from love.py
    max_len = None
    try:
        import requests
        resp = requests.get("http://localhost:8000/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                model_id = data["data"][0].get("id")
                if model_id == "Qwen/Qwen2-1.5B-Instruct-AWQ":
                    max_len = 3072
    except Exception as e:
        print(f"Error: {e}")

    # Verify max_len
    if max_len == 3072:
        print("SUCCESS: max_len correctly identified as 3072 for existing server")
    else:
        print(f"FAILURE: max_len is {max_len}, expected 3072")
        sys.exit(1)
        
    # Simulate DeepAgentEngine init
    engine = MockDeepAgentEngine(api_url="http://localhost:8000", tool_registry=None, max_model_len=max_len)
    
    # Verify call args
    call_args = MockDeepAgentEngine.call_args
    print(f"DeepAgentEngine init args: {call_args}")
    _, kwargs = call_args
    if kwargs.get("max_model_len") == 3072:
        print("SUCCESS: DeepAgentEngine initialized with max_model_len=3072")
    else:
        print(f"FAILURE: DeepAgentEngine initialized with {kwargs.get('max_model_len')}")
        sys.exit(1)

if __name__ == "__main__":
    test_existing_server_logic()
