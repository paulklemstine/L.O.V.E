import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from core.decentralized_wealth_protocol import DecentralizedWealthProtocol
from core.graph_manager import GraphDataManager
from core.signal_evolution_matrix import SignalEvolutionMatrix

class TestDecentralizedWealthProtocol(unittest.TestCase):

    def setUp(self):
        """Set up a mock GraphDataManager and SignalEvolutionMatrix for testing."""
        self.mock_kg = MagicMock(spec=GraphDataManager)
        self.mock_signal_matrix = MagicMock(spec=SignalEvolutionMatrix)
        # Mock the async method generate_signals
        self.mock_signal_matrix.generate_signals = AsyncMock()
        self.engine = DecentralizedWealthProtocol(self.mock_kg, self.mock_signal_matrix)

    def test_generate_strategies_with_no_signals(self):
        """Test that no strategies are generated when the signal matrix is empty."""
        self.mock_signal_matrix.generate_signals.return_value = []
        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 0)

    def test_generate_strategies_with_one_signal(self):
        """Test that a strategy is generated for a single signal."""
        self.mock_signal_matrix.generate_signals.return_value = [
            {
                "signal_id": "TEST_SIGNAL_001",
                "description": "Test signal.",
                "actions": ["Do something."],
                "confidence": 0.9
            }
        ]
        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]['signal_id'], 'TEST_SIGNAL_001')

if __name__ == '__main__':
    unittest.main()