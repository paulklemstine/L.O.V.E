import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before importing core.multiplayer
sys.modules["aioipfs"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Mock core.llm_api to avoid importing it and its dependencies
mock_llm_api = MagicMock()
mock_llm_api.run_llm = AsyncMock(return_value={"result": "merged"})
sys.modules["core.llm_api"] = mock_llm_api

try:
    import networkx as nx
except ImportError:
    sys.modules["networkx"] = MagicMock()
    nx = MagicMock()

# Import module under test
from core.multiplayer import MultiplayerManager

# Helper for async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper

class MockAioIpfsClient:
    def __init__(self):
        self.pubsub = MagicMock()
        self.pubsub.pub = AsyncMock()
        self.pubsub.sub = MagicMock()

    async def add(self, path):
        return {'Hash': 'QmTestCID'}
    
    async def get(self, cid, dst):
        # Create a dummy graphml file
        # We need to ensure dst directory exists or just write to it
        # Since we are mocking networkx in the manager, we might not need a real file 
        # if we also mock read_graphml.
        # But let's write a dummy file just in case.
        with open(dst, 'w') as f:
            f.write("<graphml>dummy</graphml>")
        return

    async def close(self):
        pass

class TestMultiplayerManager(unittest.TestCase):
    def setUp(self):
        self.console = MagicMock()
        self.kb = MagicMock()
        self.kb.graph = MagicMock()
        self.kb.get_node.return_value = None
        self.kb.add_node = MagicMock()
        self.kb.add_edge = MagicMock()
        
        self.ipfs_manager = MagicMock()
        self.love_state = {"version_name": "TestVersion", "hardware": {"gpu_detected": False}}
        
        self.manager = MultiplayerManager(self.console, self.kb, self.ipfs_manager, self.love_state)

    def test_initialization(self):
        self.assertEqual(self.manager.topic, "love-lobby")
        self.assertIsNotNone(self.manager.peer_id)

    @patch('core.multiplayer.aioipfs')
    def test_publish_knowledge(self, mock_aioipfs):
        # Setup mock client
        mock_client = MockAioIpfsClient()
        self.manager._get_client = AsyncMock(return_value=mock_client)
        
        # Run async test
        cid = asyncio.run(self.manager.publish_knowledge("Test export"))
        
        self.assertEqual(cid, "QmTestCID")
        # Verify pubsub.pub was called
        mock_client.pubsub.pub.assert_called_once()

    @patch('core.multiplayer.aioipfs')
    @patch('core.multiplayer.nx')
    def test_sync_knowledge(self, mock_nx, mock_aioipfs):
        # Setup mock client
        mock_client = MockAioIpfsClient()
        self.manager._get_client = AsyncMock(return_value=mock_client)
        
        # Mock nx.read_graphml to return a graph with some nodes
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [("node1", {"node_type": "test", "data": "value"})]
        mock_graph.edges.return_value = []
        mock_nx.read_graphml.return_value = mock_graph
        
        # Run async test
        result = asyncio.run(self.manager.sync_knowledge("QmRemoteCID"))
        
        self.assertTrue(result)
        # Verify node was added to KB
        self.kb.add_node.assert_called()
        # Verify file was cleaned up (mock client writes it, manager removes it)
        # We can't easily check file removal since we mocked the writing/reading flow mostly.

if __name__ == '__main__':
    unittest.main()
