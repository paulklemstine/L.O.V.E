import unittest
import os
from unittest.mock import patch, MagicMock
from core.ethereum.monitoring import monitor_and_store_balance, get_eth_balance, get_erc20_token_transfers, get_erc20_balance_for_token
from core.knowledge_graph.graph import KnowledgeGraph

class TestEthereumMonitoring(unittest.TestCase):

    def setUp(self):
        self.knowledge_graph = KnowledgeGraph(":memory:")
        self.address = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"

    @patch('requests.get')
    @patch('core.ethereum.monitoring.ETHERSCAN_API_KEY', "test_key")
    def test_get_eth_balance_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "1", "result": "1000000000000000000"}
        mock_get.return_value = mock_response

        balance = get_eth_balance(self.address)
        self.assertEqual(balance, 1.0)

    @patch('requests.get')
    @patch('core.ethereum.monitoring.ETHERSCAN_API_KEY', "test_key")
    def test_get_erc20_token_transfers_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "result": [
                {
                    "contractAddress": "0xToken1",
                    "tokenSymbol": "TKN1",
                    "tokenDecimal": "18"
                }
            ]
        }
        mock_get.return_value = mock_response

        transfers = get_erc20_token_transfers(self.address)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0]['tokenSymbol'], 'TKN1')

    @patch('requests.get')
    @patch('core.ethereum.monitoring.ETHERSCAN_API_KEY', "test_key")
    def test_get_erc20_balance_for_token_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "1", "result": "5000000000000000000"}
        mock_get.return_value = mock_response

        balance = get_erc20_balance_for_token(self.address, "0xToken1")
        self.assertEqual(balance, 5000000000000000000)

    @patch('core.ethereum.monitoring.get_eth_balance')
    @patch('core.ethereum.monitoring.get_erc20_token_transfers')
    @patch('core.ethereum.monitoring.get_erc20_balance_for_token')
    @patch('core.ethereum.monitoring.ETHERSCAN_API_KEY', "test_key")
    def test_monitor_and_store_balance(self, mock_get_erc20_balance, mock_get_transfers, mock_get_eth):
        mock_get_eth.return_value = 1.5
        mock_get_transfers.return_value = [
            {
                "contractAddress": "0xToken1",
                "tokenSymbol": "TKN1",
                "tokenDecimal": "18"
            }
        ]
        mock_get_erc20_balance.return_value = 2500000000000000000

        monitor_and_store_balance(self.address, self.knowledge_graph)

        triples = self.knowledge_graph.get_triples()
        self.assertIn((self.address, "has_eth_balance", "1.5"), triples)
        self.assertIn((self.address, "has_TKN1_balance", "2.5"), triples)

if __name__ == '__main__':
    unittest.main()