import unittest
import asyncio
from unittest.mock import patch, MagicMock

from core.agents.orchestrator import Orchestrator
from core.graph_manager import GraphDataManager

from core.memory.memory_manager import MemoryManager

class TestPhase1Integration(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        mock_graph_manager = MagicMock(spec=GraphDataManager)
        mock_memory_manager = MemoryManager(mock_graph_manager)
        self.orchestrator = Orchestrator(mock_memory_manager)

    def test_orchestrator_initialization(self):
        """
        Tests if the Orchestrator and its key components are initialized correctly.
        """
        print("\n--- Running test_orchestrator_initialization ---")
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.specialist_registry)
        self.assertIsNotNone(self.orchestrator.metacognition_agent)


if __name__ == '__main__':
    unittest.main()