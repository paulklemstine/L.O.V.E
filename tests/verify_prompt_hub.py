
import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

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

    def test_local_fallback_duplicate(self):
        """Test that it falls back to local prompts if remote disabled or fails."""
        os.environ["USE_REMOTE_PROMPTS"] = "false"
        prompt = self.registry.get_prompt("deep_agent_system")
        self.assertTrue(prompt is not None, "Should load local prompt")
        self.assertIn("L.O.V.E.", prompt, "Should contain expected content")

    
    def test_get_hub_prompt_specific(self):
        """Test specific get_hub_prompt method."""
        try:
             from langchain import hub
        except ImportError:
             return

        with patch('langchain.hub.pull') as mock_pull:
            # Mock the remote prompt object
            mock_prompt = MagicMock()
            mock_prompt.template = "REMOTE CONTENT SPECIFIC"
            mock_pull.return_value = mock_prompt
            
            # Explicit call
            result = self.registry.get_hub_prompt("hwchase17/react")
            
            mock_pull.assert_called_with("hwchase17/react")
            self.assertEqual(result, "REMOTE CONTENT SPECIFIC")

    def test_push_to_hub(self):
        """Test pushing to hub."""
        try:
             from langchain import hub
        except ImportError:
             return

        with patch('langchain.hub.push') as mock_push:
            success = self.registry.push_to_hub("my/repo", "Test Prompt")
            self.assertTrue(success)
            mock_push.assert_called()

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
            # Should not call again because it hits cache
            mock_pull.assert_called_once()

    def test_reload_clears_cache(self):
        """Test that reload clears the cache."""
        self.registry._remote_cache["test_key"] = "cached"
        self.registry.reload()
        self.assertNotIn("test_key", self.registry._remote_cache)


class TestMetacognitionHubIntegration(unittest.TestCase):
    """Tests for MetacognitionAgent's push_to_hub integration."""
    
    def setUp(self):
        PromptRegistry._instance = None
    
    def test_push_evolution_prompt(self):
        """Test that MetacognitionAgent can push prompts to Hub."""
        try:
            from langchain import hub
        except ImportError:
            print("Skipping - langchainhub not installed")
            return
            
        from core.agents.metacognition_agent import MetacognitionAgent
        
        with patch('langchain.hub.push') as mock_push:
            # Create mock memory manager
            mock_mm = MagicMock()
            mock_mm.add_episode = AsyncMock()
            
            agent = MetacognitionAgent(mock_mm)
            
            # Run the async method
            result = asyncio.run(agent.push_evolution_prompt(
                "love-agent/test-prompt",
                "This is an optimized prompt content"
            ))
            
            # Verify hub.push was called
            mock_push.assert_called()
            
            # Verify the event was recorded
            mock_mm.add_episode.assert_called()


class TestAnalystAgentHubIntegration(unittest.TestCase):
    """Tests for AnalystAgent's Hub prompt integration."""
    
    def setUp(self):
        PromptRegistry._instance = None
    
    def test_code_refactoring_fallback(self):
        """Test that code refactoring uses local fallback when Hub unavailable."""
        os.environ["USE_REMOTE_PROMPTS"] = "false"
        
        from core.agents.analyst_agent import AnalystAgent
        
        # Create agent without Hub access
        agent = AnalystAgent(memory_manager=None)
        
        # Verify the local prompt exists
        prompt = agent.registry.get_prompt("code_refactoring")
        self.assertIsNotNone(prompt, "Local code_refactoring prompt should exist")
        self.assertIn("Clean Code", prompt, "Should contain Clean Code principles")
    
    def test_security_auditor_fallback(self):
        """Test that security auditor uses local fallback when Hub unavailable."""
        os.environ["USE_REMOTE_PROMPTS"] = "false"
        
        from core.agents.analyst_agent import AnalystAgent
        
        agent = AnalystAgent(memory_manager=None)
        
        # Verify the local prompt exists
        prompt = agent.registry.get_prompt("security_auditor")
        self.assertIsNotNone(prompt, "Local security_auditor prompt should exist")
        self.assertIn("vulnerability", prompt.lower(), "Should contain vulnerability info")
    
    def test_analyst_has_new_task_types(self):
        """Test that AnalystAgent supports new task types."""
        from core.agents.analyst_agent import AnalystAgent
        
        agent = AnalystAgent(memory_manager=None)
        
        # Verify methods exist
        self.assertTrue(hasattr(agent, '_analyze_code_refactoring'))
        self.assertTrue(hasattr(agent, '_analyze_security'))


class TestHubFallbackBehavior(unittest.TestCase):
    """Tests for graceful fallback when langchainhub is not installed."""
    
    def test_hub_none_fallback(self):
        """Test behavior when hub module is None."""
        PromptRegistry._instance = None
        registry = PromptRegistry()
        
        # Even if hub is None, get_hub_prompt should return fallback
        result = registry.get_hub_prompt("nonexistent/prompt")
        
        # Should return empty string or fallback, not raise
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()

