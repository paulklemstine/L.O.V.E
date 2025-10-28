import unittest
from unittest.mock import MagicMock

from core.financial_strategy_engine import FinancialStrategyEngine
from core.graph_manager import GraphDataManager

class TestFinancialStrategyEngine(unittest.TestCase):

    def setUp(self):
        """Set up a mock GraphDataManager for testing."""
        self.mock_kg = GraphDataManager()
        self.mock_kg.add_relation = MagicMock()
        self.mock_kg.get_triples = MagicMock()
        self.engine = FinancialStrategyEngine(self.mock_kg)

    def test_generate_strategies_with_no_data(self):
        """Test that no strategies are generated when the KG is empty."""
        self.mock_kg.get_triples.return_value = []
        strategies = self.engine.generate_strategies()
        self.assertEqual(len(strategies), 0)

    def test_generate_strategies_with_portfolio_data(self):
        """Test that a portfolio analysis strategy is generated."""
        self.mock_kg.get_triples.return_value = [
            ("0x419CA6f5b6F795604938054c951c94d8629AE5Ed", "has_eth_balance", "10.5"),
            ("0x419CA6f5b6F795604938054c951c94d8629AE5Ed", "has_USDC_balance", "10000")
        ]
        strategies = self.engine.generate_strategies()
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]['strategy_id'], 'PORTFOLIO_DIVERSIFICATION_01')

    def test_generate_strategies_with_growth_token_data(self):
        """Test that a growth token strategy is generated."""
        self.mock_kg.get_triples.return_value = [
            ("NEW_TOKEN", "listed_on", "Binance")
        ]
        strategies = self.engine.generate_strategies()
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]['strategy_id'], 'INVEST_IN_NEW_TOKEN')

if __name__ == '__main__':
    unittest.main()