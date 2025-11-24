import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import os
import faiss

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

    async def asyncSetUp(self):
        """Set up a new MemoryManager instance for each test."""
        # Reset mocks before each test to ensure isolation
        mock_graph_manager.reset_mock()

        # Patch the sentence transformer model to avoid loading it in every test
        self.mock_sentence_transformer = MagicMock()
        self.mock_sentence_transformer.encode.return_value = np.random.rand(1, 384)

        with patch('sentence_transformers.SentenceTransformer', return_value=self.mock_sentence_transformer):
            self.memory_manager = await MemoryManager.create(graph_data_manager=mock_graph_manager)

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

class TestMemoryManagerSemanticSearch(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        """
        Set up for semantic search tests. This now includes pre-populating the
        mock graph manager so the async `create` method can build a trained index.
        """
        self.mock_graph_manager = MagicMock(spec=GraphDataManager)
        # Now, create the memory manager, which will trigger the async index build
        self.memory_manager = await MemoryManager.create(graph_data_manager=self.mock_graph_manager)

        # For the test, manually create a simple, trained index
        self.memory_manager.faiss_index = faiss.IndexFlatL2(self.memory_manager.faiss_dimension)

        # Clean up any old index files
        if os.path.exists(self.memory_manager.faiss_index_path):
            os.remove(self.memory_manager.faiss_index_path)
        if os.path.exists(self.memory_manager.faiss_id_map_path):
            os.remove(self.memory_manager.faiss_id_map_path)

    def tearDown(self):
        """Clean up FAISS index files after tests."""
        if os.path.exists(self.memory_manager.faiss_index_path):
            os.remove(self.memory_manager.faiss_index_path)
        if os.path.exists(self.memory_manager.faiss_id_map_path):
            os.remove(self.memory_manager.faiss_id_map_path)

    async def test_semantic_search_finds_related_memories(self):
        """
        Verify that _find_and_link_related_memories uses FAISS to find semantically
        similar notes and passes them as candidates to the LLM.
        """
        # --- Arrange ---
        # --- Arrange ---
        # 1. Create memory notes with known semantic relationships
        note1 = MemoryNote(content="The cat sat on the mat.", embedding=self.memory_manager.embedding_model.encode(["The cat sat on the mat."])[0], contextual_description="", keywords=[], tags=[])
        note2 = MemoryNote(content="A feline was resting on the rug.", embedding=self.memory_manager.embedding_model.encode(["A feline was resting on the rug."])[0], contextual_description="", keywords=[], tags=[])
        note3 = MemoryNote(content="The dog barked at the moon.", embedding=self.memory_manager.embedding_model.encode(["The dog barked at the moon."])[0], contextual_description="", keywords=[], tags=[])

        # 2. Manually populate the memory manager's state
        self.memory_manager.faiss_index.add(np.array([note1.embedding, note2.embedding, note3.embedding], dtype=np.float32))
        self.memory_manager.faiss_id_map.extend([note1.id, note2.id, note3.id])

        # Mock the graph manager to return the notes when queried
        self.mock_graph_manager.get_node.side_effect = lambda node_id: {
            note1.id: note1.to_node_attributes(),
            note2.id: note2.to_node_attributes(),
            note3.id: note3.to_node_attributes(),
        }.get(node_id)

        # 3. Create the new note to be linked
        new_note_content = "There was a kitty on the carpet."
        new_note = MemoryNote(
            content=new_note_content,
            embedding=self.memory_manager.embedding_model.encode([new_note_content])[0],
            contextual_description="", keywords=[], tags=[]
        )
        # Add the new note to the index to ensure ntotal > 1
        self.memory_manager.faiss_index.add(np.array([new_note.embedding], dtype=np.float32))
        self.memory_manager.faiss_id_map.append(new_note.id)

        # 4. Mock the LLM call to isolate the candidate selection logic
        mock_run_llm = AsyncMock(return_value={"result": "[]"})

        # --- Act ---
        with patch('core.llm_api.run_llm', new=mock_run_llm):
            await self.memory_manager._find_and_link_related_memories(new_note, top_k=2)

        # --- Assert ---
        # 1. Verify that the LLM was called
        mock_run_llm.assert_called_once()

        # 2. Extract the prompt sent to the LLM
        prompt = mock_run_llm.call_args[0][0]

        # 3. Check that the prompt contains the summaries of the correct candidates
        self.assertIn(note1.id, prompt)
        self.assertIn(note1.content, prompt)
        self.assertIn(note2.id, prompt)
        self.assertIn(note2.content, prompt)

        # 4. Check that the prompt does NOT contain the dissimilar note
        self.assertNotIn(note3.id, prompt)
        self.assertNotIn(note3.content, prompt)


if __name__ == '__main__':
    unittest.main()
