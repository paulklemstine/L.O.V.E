
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompt_registry import PromptRegistry

class TestPromptHub(unittest.TestCase):
    def setUp(self):
        # Reset singleton for clean test
        PromptRegistry._instance = None
        self.registry = PromptRegistry()
        
    def test_local_fallback(self):
        """Test that it falls back to local prompts if remote disabled or fails."""
        os.environ["USE_REMOTE_PROMPTS"] = "false"
        prompt = self.registry.get_prompt("deep_agent_system")
        self.assertTrue(prompt is not None, "Should load local prompt")
        self.assertIn("L.O.V.E.", prompt, "Should contain expected content")

    def test_local_fallback(self):
        """Test that it falls back to local prompts if remote disabled or fails."""
        os.environ["USE_REMOTE_PROMPTS"] = "false"
        prompt = self.registry.get_prompt("deep_agent_system")
        self.assertTrue(prompt is not None, "Should load local prompt")
        self.assertIn("L.O.V.E.", prompt, "Should contain expected content")

    def test_remote_pull(self):
        """Test that it pulls from hub if enabled (only if module exists)."""
        try:
             from langchain import hub
        except ImportError:
             print("Skipping remote pull test - langchain.hub not found")
             return

        with patch('langchain.hub.pull') as mock_pull:
            os.environ["USE_REMOTE_PROMPTS"] = "true"
            os.environ["LANGCHAIN_HUB_REPO"] = "love-agent"
            
            # Mock the remote prompt object
            mock_prompt = MagicMock()
            mock_prompt.template = "REMOTE PROMPT CONTENT"
            mock_pull.return_value = mock_prompt
            
            # Get prompt
            prompt = self.registry.get_prompt("deep_agent_system")
            
            # Verify
            mock_pull.assert_called_with("love-agent/deep_agent_system")
            self.assertEqual(prompt, "REMOTE PROMPT CONTENT")
            
            # Verify Cache
            self.registry.get_prompt("deep_agent_system")
            mock_pull.assert_called_once() # Should not call again

    def test_reload_clears_cache(self):
        """Test that reload clears the cache."""
        self.registry._remote_cache["test_key"] = "cached"
        self.registry.reload()
        self.assertNotIn("test_key", self.registry._remote_cache)

if __name__ == '__main__':
    unittest.main()
