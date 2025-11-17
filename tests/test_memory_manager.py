import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

# Mock the GraphDataManager before importing MemoryManager
from core.graph_manager import GraphDataManager
mock_graph_manager = MagicMock(spec=GraphDataManager)

from core.memory.memory_manager import MemoryManager, MemoryNote

class TestMemoryNote(unittest.TestCase):
    """Test suite for the MemoryNote dataclass."""

    def test_serialization_with_valid_embedding(self):
        """Verify that a MemoryNote with a numpy embedding is serialized correctly."""
        # --- Arrange ---
        note = MemoryNote(
            content="Test content",
            embedding=np.array([0.1, 0.2, 0.3]),
            contextual_description="A test note.",
            keywords=["test", "numpy"],
            tags=["Serialization"],
        )

        # --- Act ---
        attributes = note.to_node_attributes()

        # --- Assert ---
        self.assertEqual(attributes['embedding'], '[0.1, 0.2, 0.3]')

    def test_serialization_with_none_embedding(self):
        """Verify that a MemoryNote with a None embedding is safely serialized."""
        # --- Arrange ---
        note = MemoryNote(
            content="Test content without embedding",
            embedding=None,
            contextual_description="A test note.",
            keywords=["test", "none"],
            tags=["Serialization"],
        )

        # --- Act ---
        attributes = note.to_node_attributes()

        # --- Assert ---
        self.assertEqual(attributes['embedding'], '[]')

    def test_deserialization_with_valid_embedding(self):
        """Verify that node attributes with a valid embedding are deserialized correctly."""
        # --- Arrange ---
        node_id = "test-node-1"
        attributes = {
            "content": "Test content",
            "embedding": '[0.1, 0.2, 0.3]',
            "contextual_description": "A test note.",
            "keywords": "test,numpy",
            "tags": "Deserialization",
        }
        expected_embedding = np.array([0.1, 0.2, 0.3])

        # --- Act ---
        note = MemoryNote.from_node_attributes(node_id, attributes)

        # --- Assert ---
        self.assertIsInstance(note.embedding, np.ndarray)
        np.testing.assert_array_equal(note.embedding, expected_embedding)

    def test_deserialization_with_empty_embedding(self):
        """Verify that node attributes with an empty embedding string are deserialized correctly."""
        # --- Arrange ---
        node_id = "test-node-2"
        attributes = {
            "content": "Test content",
            "embedding": '[]',
            "contextual_description": "A test note.",
            "keywords": "test,none",
            "tags": "Deserialization",
        }

        # --- Act ---
        note = MemoryNote.from_node_attributes(node_id, attributes)

        # --- Assert ---
        self.assertIsInstance(note.embedding, np.ndarray)
        self.assertEqual(note.embedding.size, 0)


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up a new MemoryManager instance for each test."""
        # Reset mocks before each test to ensure isolation
        mock_graph_manager.reset_mock()
        self.memory_manager = MemoryManager(graph_data_manager=mock_graph_manager)

    async def test_ingest_cognitive_cycle_creates_structured_memory(self):
        """
        Verify that ingest_cognitive_cycle correctly formats the input and
        triggers the agentic memory processing pipeline by calling add_episode.
        """
        # --- Arrange ---
        # Spy on the add_episode method to see what it's called with.
        # We don't need to mock the full pipeline, just the entry point.
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
        self.assertIn(f"{command}", actual_content)
        self.assertIn(output, actual_content)
        self.assertIn(reasoning_prompt, actual_content)

        # 4. Verify that the correct tags were passed
        self.assertIn('CognitiveCycle', actual_tags)

if __name__ == '__main__':
    unittest.main()
