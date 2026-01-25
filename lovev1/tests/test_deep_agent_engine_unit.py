import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Ensure core module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.deep_agent_engine import DeepAgentEngine
from core.tool_registry import ToolRegistry

class TestDeepAgentEngineUnit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool_registry = ToolRegistry()
        # Register a dummy tool
        def dummy_tool(arg1: str):
            """A dummy tool for testing."""
            return f"Processed {arg1}"
        self.tool_registry.register(dummy_tool, name="dummy_tool")
        
        self.engine = DeepAgentEngine(
            api_url="http://localhost:8000",
            tool_registry=self.tool_registry,
            use_pool=False
        )

    def test_adapt_tools_for_deepagent(self):
        """Verify that tools from the registry are correctly formatted for the prompt."""
        formatted_tools = self.engine._adapt_tools_for_deepagent()
        self.assertIn("Tool Name: `dummy_tool`", formatted_tools)
        self.assertIn("Description: A dummy tool for testing.", formatted_tools)
        self.assertIn("Arguments JSON Schema:", formatted_tools)

    @patch('httpx.AsyncClient')
    async def test_generate_raw_vllm_integration(self, mock_client_cls):
        """Verify that generate_raw makes the correct API call to vLLM."""
        # Setup mock response
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Test response"}}
            ]
        }
        mock_client.post.return_value = mock_response

        # Execute
        prompt = "test prompt"
        response = await self.engine.generate_raw(prompt)

        # Verify
        self.assertEqual(response, "Test response")
        
        # Check API call details
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        url = call_args[0][0]
        kwargs = call_args[1]
        
        self.assertEqual(url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(kwargs['json']['messages'][0]['content'], prompt)
        # Verify default model params are present (or whatever defaults DeepAgentEngine uses)
        self.assertIn('model', kwargs['json'])

if __name__ == '__main__':
    unittest.main()
