
import unittest
import sys
import os

# Ensure core module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompt_registry import PromptRegistry

class TestPromptRegistryVerification(unittest.TestCase):
    def setUp(self):
        self.registry = PromptRegistry()
        self.registry.reload()

    def test_deep_agent_prompt_interpolation(self):
        """
        Verify that deep_agent_system renders the 'prompt' variable.
        """
        test_prompt = "TEST_PROMPT_INJECTION_SUCCESS_12345"
        
        rendered = self.registry.render_prompt(
            "deep_agent_system",
            persona_json="{}",
            tools_desc="",
            kb_context="",
            prompt=test_prompt
        )
        
        self.assertIn(test_prompt, rendered)
        self.assertIn("### CURRENT TASK", rendered)
        print("\n[SUCCESS] deep_agent_system correctly interpolates the user prompt.")

if __name__ == '__main__':
    unittest.main()
