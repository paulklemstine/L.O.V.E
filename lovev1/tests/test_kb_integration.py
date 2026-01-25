import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.kb_tools import query_knowledge_base, search_memories, get_kb_summary
from core.deep_agent_engine import DeepAgentEngine

class TestKBIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_kb = MagicMock()
        self.mock_memory = MagicMock()
        self.mock_tool_registry = MagicMock()
        
    def test_query_knowledge_base(self):
        # Setup mock return
        self.mock_kb.query_nodes.return_value = ["task1", "task2"]
        
        def get_node_side_effect(node_id):
            if node_id == "task1":
                return {"id": "task1", "type": "task", "status": "active"}
            elif node_id == "task2":
                return {"id": "task2", "type": "task", "status": "pending"}
            return None
            
        self.mock_kb.get_node.side_effect = get_node_side_effect
        
        # Test
        result = query_knowledge_base("task", 5, self.mock_kb)
        data = json.loads(result)
        
        # Verify
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["node_type"], "task")
        self.assertEqual(data["results"][0]["id"], "task1")
        
    def test_search_memories(self):
        # Setup mock return
        self.mock_memory.retrieve_relevant_folded_memories.return_value = [
            {"content": "memory1", "score": 0.9},
            {"content": "memory2", "score": 0.8}
        ]
        
        # Test
        result = search_memories("test query", 2, self.mock_memory)
        data = json.loads(result)
        
        # Verify
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["memories"][0]["content"], "memory1")
        
    def test_get_kb_summary(self):
        # Setup mock return
        self.mock_kb.summarize_graph.return_value = ("Graph Summary", {"nodes": 10})
        
        # Test
        result = get_kb_summary(self.mock_kb)
        
        # Verify
        self.assertEqual(result, "Graph Summary")
        
    def test_deep_agent_context_injection(self):
        # Setup DeepAgent with mocks
        agent = DeepAgentEngine(
            api_url="http://test",
            tool_registry=self.mock_tool_registry,
            knowledge_base=self.mock_kb,
            memory_manager=self.mock_memory
        )
        
        # Mock KB tools calls since they are imported inside the method
        with patch('core.kb_tools.get_kb_summary') as mock_get_summary, \
             patch('core.kb_tools.search_memories') as mock_search:
            
            mock_get_summary.return_value = "Mock KB Context"
            mock_search.return_value = json.dumps({"count": 1, "memories": ["Mock Memory"]})
            
            # Test
            context = agent._get_kb_context("test prompt")
            
            # Verify
            self.assertIn("Mock KB Context", context)
            self.assertIn("Mock Memory", context)
            self.assertIn("📚 Knowledge Base Context:", context)
            self.assertIn("🧠 Relevant Past Experiences:", context)

    def test_memory_manager_attribute_fix(self):
        """Regression test for AttributeError: 'MemoryManager' object has no attribute 'model'"""
        from core.memory.memory_manager import MemoryManager
        
        # Mock dependencies
        mock_graph_manager = MagicMock()
        mock_graph_manager.query_nodes.return_value = [] # Return empty to avoid further processing
        
        # Instantiate MemoryManager with mocks
        # We need to patch SentenceTransformer to avoid loading the real model
        with patch('core.memory.memory_manager.SentenceTransformer') as MockTransformer:
            mock_model = MagicMock()
            MockTransformer.return_value = mock_model
            
            # We also need to mock os.path.exists to avoid loading FAISS from disk
            with patch('os.path.exists', return_value=False):
                manager = MemoryManager(mock_graph_manager)
                
                # Verify initialization
                self.assertEqual(manager.embedding_model, mock_model)
                
                # Call the method that was failing
                manager.retrieve_relevant_folded_memories("test query")
                
                # Verify it used embedding_model, not model
                mock_model.encode.assert_called_with("test query")

if __name__ == '__main__':
    unittest.main()
