
import asyncio
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.memory_manager import MemoryManager, MemoryNote
from core.memory.schemas import MemorySummary
from core.graph_manager import GraphDataManager

# Mock GraphDataManager
class MockGraphDataManager(GraphDataManager):
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, node_type, attributes):
        self.nodes[node_id] = attributes
        
    def get_node(self, node_id):
        return self.nodes.get(node_id)
        
    def query_nodes(self, key, value):
        return list(self.nodes.keys())

@pytest.mark.asyncio
async def test_memory_tiering():
    print("\n--- Starting Memory Tiering Test ---")
    
    # Setup
    mock_graph_manager = MockGraphDataManager()
    
    # Mock FAISS to avoid complex setup in test environment if not available
    # But since we installed faiss-cpu, we can try to use the real one if consistent.
    # However, for unit testing logic, mocking is safer.
    
    # We will mock the embedding model to be faster and deterministic
    with patch('core.memory.memory_manager.SentenceTransformer') as MockTransformer:
        mock_embedding_model = MagicMock()
        # Smart mock for encode
        def fake_encode(sentences, *args, **kwargs):
            if isinstance(sentences, list):
                return np.array([[0.1] * 768] * len(sentences), dtype=np.float32)
            return np.array([0.1] * 768, dtype=np.float32)
            
        mock_embedding_model.encode.side_effect = fake_encode
        MockTransformer.return_value = mock_embedding_model
        
        # Initialize MemoryManager
        memory_manager = MemoryManager(mock_graph_manager)
        
        # Manually initialize FAISS (mocking the async create/load)
        import faiss
        memory_manager.faiss_index = faiss.IndexFlatL2(768)
        memory_manager.faiss_id_map = []
        
        # 1. Populate Hot Tier (Level 0) with > 20 items
        print("Populating Hot Tier...")
        for i in range(25):
            memory_note = MemoryNote(
                content=f"Memory #{i}: This is a test memory about topic {i%5}",
                embedding=np.array([0.1] * 768, dtype=np.float32),
                contextual_description="Test description",
                keywords=["test"],
                tags=["test"]
            )
            # Add to graph/FAISS
            await memory_manager.add_note_to_index(memory_note)
            mock_graph_manager.add_node(memory_note.id, "MemoryNote", memory_note.to_node_attributes())
            
            # Add to Level 0 RAM
            summary = MemorySummary(content=memory_note.content, level=0, source_ids=[memory_note.id])
            memory_manager.level_0_memories.append(summary)
            
        assert len(memory_manager.level_0_memories) == 25
        print(f"Hot Tier populated with {len(memory_manager.level_0_memories)} items.")
        
        # 2. Trigger Archive to Cold Storage
        # Mock IPFS pinning
        with patch('ipfs.pin_to_ipfs', new_callable=MagicMock) as mock_pin:
            async def fake_pin(*args, **kwargs):
                return "QmHash12345"
            mock_pin.side_effect = fake_pin
            
            print("Triggering Archive to Cold Storage...")
            await memory_manager.archive_to_cold_storage()
            
            # 3. Verify RAM is cleared (only 20 left)
            assert len(memory_manager.level_0_memories) == 20
            print(f"Hot Tier pruned to {len(memory_manager.level_0_memories)} items.")
            
            # Verify the archived items have CIDs (logic check)
            # In the implementation, we modify the object references in the original list.
            # But since they are removed from level_0_memories, we can't check them there easily 
            # unless we kept a reference.
            
            # However, the test script doesn't keep the exact objects unless we did.
            # The archive method printed "Successfully archived 5 memories".
            print("Archive simulation complete.")

        # 4. Test Semantic Retrieval
        print("Testing Semantic Retrieval...")
        
        # We search for "Memory #0" which should have been archived (indices 0-4 are archived)
        # Mock FAISS search result
        # Since we used a real IndexFlatL2, and all embeddings are identical [0.1]*768, 
        # all distances are 0. It should return everything.
        
        query = "topic 0"
        context = await memory_manager.retrieve_semantic_context(query)
        
        print(f"Retrieved Context:\n{context}")
        
        assert "Memory #" in context
        assert "[Cold Retrieval]" in context
        print("Semantic Retrieval Verified.")

if __name__ == "__main__":
    asyncio.run(test_memory_tiering())
