import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

# Mock the GraphDataManager before importing MemoryManager
from core.graph_manager import GraphDataManager
mock_graph_manager = MagicMock(spec=GraphDataManager)

from core.memory.memory_manager import MemoryManager, MemoryNote

class TestMemoryManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up a new MemoryManager instance for each test."""
        # Reset mocks before each test to ensure isolation
        mock_graph_manager.reset_mock()
        self.memory_manager = MemoryManager(graph_data_manager=mock_graph_manager)

    @patch('core.memory.memory_manager.run_llm', new_callable=AsyncMock)
    @patch('core.llm_api.run_llm', new_callable=AsyncMock)
    async def test_ingest_cognitive_cycle_creates_structured_memory(self, mock_sentence_transformer, mock_run_llm):
        """
        Verify that ingest_cognitive_cycle correctly formats the input and
        triggers the agentic memory processing pipeline.
        """
        # --- Arrange ---
        # Mock the SentenceTransformer to return a predictable embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sentence_transformer.return_value = mock_model

        # Mock the LLM call to return a predictable structured response
        mock_run_llm.return_value = {
            "result": '{"contextual_description": "Test description", "keywords": ["test", "cycle"], "tags": ["Testing"]}'
        }

        # Use a spy to capture the content passed to add_episode
        add_episode_spy = AsyncMock()
        self.memory_manager.add_episode = add_episode_spy

        command = "test_command --arg1"
        output = "Command executed successfully."
        reasoning_prompt = "I decided to run the test command."

        # --- Act ---
        await self.memory_manager.ingest_cognitive_cycle(command, output, reasoning_prompt)

        # --- Assert ---
        # 1. Check that add_episode was called exactly once
        add_episode_spy.assert_called_once()

        # 2. Extract the arguments passed to the spy
        call_args = add_episode_spy.call_args
        actual_content = call_args[0][0]
        actual_tags = call_args[1].get('tags')


        # 3. Verify the structure of the ingested content
        self.assertIn("Cognitive Event: Agent decided to act.", actual_content)
        self.assertIn(f"CMD: {command}", actual_content)
        self.assertIn(output, actual_content)
        self.assertIn(reasoning_prompt, actual_content)

        # 4. Verify that the correct tags were passed
        self.assertIn('CognitiveCycle', actual_tags)

if __name__ == '__main__':
    unittest.main()
