
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
import json

# Ensure core module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disable FUSE Harness to prevent initialization hang
os.environ["ENABLE_FUSE_HARNESS"] = "false"

from core.deep_agent_engine import DeepAgentEngine
from core.prompt_registry import PromptRegistry

class TestDeepAgentPromptVerification(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # We need to make sure we're using the REAL PromptRegistry to load the yaml
        # but we want to intercept the LLM call.
        self.registry = PromptRegistry()
        # Force reload to ensure we get the latest file changes
        self.registry.reload()
        
        self.engine = DeepAgentEngine(
            api_url="http://localhost:8000",
            max_model_len=8192
        )

    @patch('httpx.AsyncClient')
    async def test_deep_agent_system_prompt_structure(self, mock_client_cls):
        """
        Verify that the 'deep_agent_system' prompt is used and
        that the user's prompt is correctly injected into it.
        """
        # Setup mock response for httpx
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "{\"status\": \"ok\", \"thought\": \"Mocked thought\", \"action\": null}"}}
            ]
        }
        mock_client.post.return_value = mock_response

        # The Prompt we want the agent to run
        test_prompt = "CRITICAL_TEST_TOKEN: Please output JSON format."
        
        # Run the engine
        # We don't care about the return value as much as the request payload
        await self.engine.run(test_prompt)
        
        # Verify the call was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        kwargs = call_args[1]
        payload = kwargs['json']
        messages = payload['messages']
        
        # Find the system prompt (which DeepAgentEngine.run might inject as the first message 
        # OR it renders it into the user message if it doesn't separate them strictly.
        # DeepAgent logic: 
        # system_prompt = registry.render_prompt(..., prompt=prompt) 
        # payload = { messages: [{role: user, content: system_prompt}] }
        # So we expect ONE user message containing the rendered system prompt.
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        sent_content = messages[0]['content']
        
        # 1. Verify the static parts of the new prompt are there
        self.assertIn("### ROLE", sent_content)
        self.assertIn("### CONTEXT", sent_content)
        self.assertIn("### CURRENT TASK", sent_content)
        self.assertIn("YOU MUST COMPLY EXACTLY", sent_content)
        
        # 2. Verify the dynamic injection of the user prompt
        self.assertIn(test_prompt, sent_content)
        
        print("\n[SUCCESS] Verified that DeepAgentEngine is sending the correctly formatted system prompt.")

if __name__ == '__main__':
    unittest.main()
