import asyncio
import json
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np

# Mock dependencies before importing the module under test
# We only mock sentence_transformers here. llm_api will be handled manually.
sys_modules_patch = patch.dict('sys.modules', {
    'sentence_transformers': MagicMock(),
})
sys_modules_patch.start()

from core.memory.memory_manager import MemoryManager

class TestMemoryManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up a fresh MemoryManager instance for each test."""
        # --- Manually mock the core.llm_api module ---
        # This prevents the real module and its dependencies (like display) from being imported.
        self.mock_llm_api = MagicMock()
        self.mock_run_llm = AsyncMock()
        self.mock_llm_api.run_llm = self.mock_run_llm
        sys.modules['core.llm_api'] = self.mock_llm_api

        # We patch SentenceTransformer inside the tested module's namespace
        self.sentence_transformer_patch = patch('core.memory.memory_manager.SentenceTransformer')
        self.mock_sentence_transformer_cls = self.sentence_transformer_patch.start()
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.random.rand(384)
        self.mock_sentence_transformer_cls.return_value = self.mock_model

        # Patch the file operations to avoid actual file I/O
        self.open_patch = patch('builtins.open', unittest.mock.mock_open())
        self.mock_open = self.open_patch.start()
        self.os_path_exists_patch = patch('os.path.exists', return_value=False)
        self.mock_os_path_exists = self.os_path_exists_patch.start()
        self.nx_read_graphml_patch = patch('networkx.read_graphml', side_effect=IOError)
        self.mock_nx_read = self.nx_read_graphml_patch.start()
        self.nx_write_graphml_patch = patch('networkx.write_graphml')
        self.mock_nx_write = self.nx_write_graphml_patch.start()

        self.memory = MemoryManager()

    def tearDown(self):
        """Stop all patches and clean up sys.modules."""
        self.sentence_transformer_patch.stop()
        self.open_patch.stop()
        self.os_path_exists_patch.stop()
        self.nx_read_graphml_patch.stop()
        self.nx_write_graphml_patch.stop()

        # Important: Remove the mock from sys.modules to not affect other tests
        if 'core.llm_api' in sys.modules:
            del sys.modules['core.llm_api']

        sys_modules_patch.stop()

    async def test_add_agentic_memory_creates_node_and_links(self):
        """
        Verify that adding an episode triggers the full A-MEM pipeline:
        1. Creates a structured memory note via an LLM call.
        2. Adds the note to the graph.
        3. Attempts to link it to existing memories via a second LLM call.
        4. Triggers the memory evolution process via a third LLM call.
        """
        # --- Setup Mocks ---
        llm_response_create = {
            "contextual_description": "Test description", "keywords": ["test", "mock"], "tags": ["Testing"]
        }
        llm_response_link = {
            "links": [{"target_id": "existing-node-0", "reason": "It is a related test."}]
        }
        llm_response_evolve = {
            "updated_contextual_description": "Evolved description", "updated_keywords": ["test", "evolved"], "updated_tags": ["Testing", "Evolved"]
        }
        # Configure the side_effect on the mock from setUp
        self.mock_run_llm.side_effect = [
            json.dumps(llm_response_create),
            json.dumps(llm_response_link),
            json.dumps(llm_response_evolve)
        ]

        # --- Pre-populate Graph ---
        mock_embedding = np.random.rand(384).tolist()
        self.memory.memory_graph.add_node(
            "existing-node-0",
            content="An old memory about testing.",
            embedding=json.dumps(mock_embedding), # Use json.dumps for correct formatting
            contextual_description="An old test.",
            keywords="test,old",
            tags="Testing"
        )
        self.assertEqual(self.memory.memory_graph.number_of_nodes(), 1)

        # --- Trigger Action ---
        self.memory.add_episode("test task", "test outcome", True)

        # Allow the async tasks created in add_episode to complete
        await asyncio.sleep(0.01)

        # --- Assertions ---
        self.assertEqual(self.memory.memory_graph.number_of_nodes(), 2, "A new node should have been added")
        self.assertEqual(self.memory.memory_graph.number_of_edges(), 1, "A link should have been created")

        self.assertEqual(self.mock_run_llm.call_count, 3, "Should call LLM for create, link, and evolve")

        evolved_node = self.memory.memory_graph.nodes["existing-node-0"]
        self.assertEqual(evolved_node['contextual_description'], llm_response_evolve['updated_contextual_description'])

        self.mock_open().write.assert_called() # LTM save
        self.assertGreaterEqual(self.mock_nx_write.call_count, 2) # A-MEM save (link + evolve)

if __name__ == '__main__':
    unittest.main()
