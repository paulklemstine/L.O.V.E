import unittest
from unittest.mock import AsyncMock, patch
import asyncio
import sys
import os

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.subliminal_agent import SubliminalAgent

class TestSubliminalLogic(unittest.TestCase):
    def setUp(self):
        self.agent = SubliminalAgent()

    @patch('core.subliminal_agent.run_llm', new_callable=AsyncMock)
    def test_length_constraint_logic(self, mock_run_llm):
        # Mock LLM returning a long phrase
        mock_run_llm.return_value = {"result": "THIS IS WAY TOO LONG FOR A SUBLIMINAL PHRASE"}
        
        async def run():
            # The agent logic should truncate this
            phrase = await self.agent.generate_subliminal_phrase({}, "context")
            return phrase
            
        phrase = asyncio.run(run())
        print(f"Long input -> Output: {phrase}")
        words = phrase.split()
        self.assertLessEqual(len(words), 3, "Agent failed to truncate long phrase")

    @patch('core.subliminal_agent.run_llm', new_callable=AsyncMock)
    def test_context_aware_duplication_avoidance(self, mock_run_llm):
        # Mock LLM returning a forbidden phrase initially
        # We need to mock side_effect to return forbidden first, then valid
        mock_run_llm.side_effect = [
            {"result": "FORBIDDEN PHRASE"},  # First call returns forbidden
            {"result": "UNIQ PHRASE"}       # Second call (fallback/retry) returns distinct
        ]
        
        async def run():
            return await self.agent.generate_context_aware_subliminal(
                story_beat="test",
                forbidden_phrases=["FORBIDDEN PHRASE"]
            )
            
        phrase = asyncio.run(run())
        print(f"Forbidden input -> Output: {phrase}")
        self.assertNotEqual(phrase, "FORBIDDEN PHRASE")

if __name__ == '__main__':
    unittest.main()
