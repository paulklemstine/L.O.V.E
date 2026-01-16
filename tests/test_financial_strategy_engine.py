import unittest
import asyncio
from unittest.mock import MagicMock, patch

from core.financial_strategy_engine import FinancialStrategyEngine
from core.graph_manager import GraphDataManager

class TestFinancialStrategyEngine(unittest.TestCase):

    def setUp(self):
        """Set up a mock GraphDataManager for testing."""
        self.mock_kg = GraphDataManager()
        self.mock_kg.add_relation = MagicMock()
        self.mock_kg.get_triples = MagicMock()
        self.engine = FinancialStrategyEngine(self.mock_kg)

    @patch('core.financial_strategy_engine.fetch_new_token_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.fetch_defi_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.httpx.AsyncClient')
    def test_generate_strategies_with_no_data(self, mock_httpx, mock_fetch_defi, mock_fetch_token):
        """Test that no strategies are generated when the KG is empty."""
        self.mock_kg.get_triples.return_value = []
        # Mock the async client's get method to return a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx.return_value.__aenter__.return_value.get.return_value = mock_response

        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 0)

    @patch('core.financial_strategy_engine.fetch_new_token_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.fetch_defi_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.httpx.AsyncClient')
    def test_generate_strategies_with_portfolio_data(self, mock_httpx, mock_fetch_defi, mock_fetch_token):
        """Test that a portfolio analysis strategy is generated."""
        self.mock_kg.get_triples.return_value = [
            ("0x419CA6f5b6F795604938054c951c94d8629AE5Ed", "has_eth_balance", "10.5"),
            ("0x419CA6f5b6F795604938054c951c94d8629AE5Ed", "has_USDC_balance", "10000")
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx.return_value.__aenter__.return_value.get.return_value = mock_response

        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]['strategy_id'], 'PORTFOLIO_DIVERSIFICATION_01')

    @patch('core.financial_strategy_engine.fetch_new_token_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.fetch_defi_opportunities', return_value=[])
    @patch('core.financial_strategy_engine.httpx.AsyncClient')
    def test_generate_strategies_with_growth_token_data(self, mock_httpx, mock_fetch_defi, mock_fetch_token):
        """Test that a growth token strategy is generated."""
        self.mock_kg.get_triples.return_value = [
            ("NEW_TOKEN", "listed_on", "Binance")
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx.return_value.__aenter__.return_value.get.return_value = mock_response

        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]['strategy_id'], 'INVEST_IN_NEW_TOKEN')

    @patch('core.financial_strategy_engine.fetch_new_token_opportunities')
    @patch('core.financial_strategy_engine.fetch_defi_opportunities')
    @patch('core.financial_strategy_engine.httpx.AsyncClient')
    def test_generate_decentralized_strategies(self, mock_httpx, mock_fetch_defi, mock_fetch_token):
        """Test that decentralized strategies are generated."""
        self.mock_kg.get_triples.return_value = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx.return_value.__aenter__.return_value.get.return_value = mock_response

        # Configure mocks for decentralized opportunities
        mock_fetch_defi.return_value = [
            {
                "opportunity_id": "DEFI_STAKE_001", "platform": "ExampleYield", "asset": "ETH",
                "apy": 15.5, "type": "staking", "description": "High-yield staking for ETH.",
                "action": "Stake ETH"
            }
        ]
        mock_fetch_token.return_value = [
            {
                "opportunity_id": "NEW_TOKEN_001", "token_symbol": "AGAPE", "platform": "Uniswap",
                "description": "AGAPE token.", "reasoning": "Strong community.", "action": "Acquire AGAPE"
            }
        ]

        strategies = asyncio.run(self.engine.generate_strategies())
        self.assertEqual(len(strategies), 2)
        self.assertIn('DECENTRALIZED_DEFI_STAKE_001', [s['strategy_id'] for s in strategies])
        self.assertIn('DECENTRALIZED_NEW_TOKEN_001', [s['strategy_id'] for s in strategies])


if __name__ == '__main__':
    unittest.main()
