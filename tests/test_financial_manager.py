import unittest
import os
from unittest.mock import patch, MagicMock
from core.ethereum.financial_manager import FinancialManager
from core.knowledge_graph.graph import KnowledgeGraph

class TestFinancialManager(unittest.TestCase):

    def setUp(self):
        # Ensure a clean knowledge graph for each test
        if os.path.exists("kg.json"):
            os.remove("kg.json")
        self.knowledge_graph = KnowledgeGraph()
        self.creator_address = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"
        self.financial_manager = FinancialManager(self.knowledge_graph, self.creator_address)

    @patch('core.ethereum.monitoring.monitor_and_store_balance')
    def test_monitor_creator_address(self, mock_monitor):
        self.financial_manager.monitor_creator_address()
        mock_monitor.assert_called_once_with(self.creator_address, self.knowledge_graph)

    @patch('core.ethereum.financial_manager.list_wallets')
    @patch('core.ethereum.financial_manager.get_eth_balance')
    def test_track_internal_balances(self, mock_get_eth_balance, mock_list_wallets):
        # Make sure the mock addresses are valid checksum addresses
        mock_wallet_1 = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        mock_wallet_2 = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        mock_list_wallets.return_value = [mock_wallet_1, mock_wallet_2]
        mock_get_eth_balance.side_effect = [1.23, 4.56]

        self.financial_manager.track_internal_balances()

        triples = self.financial_manager.knowledge_graph.get_triples()
        self.assertIn((mock_wallet_1, 'has_eth_balance', '1.23'), triples)
        self.assertIn((mock_wallet_2, 'has_eth_balance', '4.56'), triples)
        mock_get_eth_balance.assert_any_call(mock_wallet_1)
        mock_get_eth_balance.assert_any_call(mock_wallet_2)

    @patch('core.ethereum.financial_manager.send_eth')
    def test_execute_eth_transaction(self, mock_send_eth):
        from_address = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        to_address = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        password = "password123"
        amount = 0.1

        self.financial_manager.execute_transaction(from_address, password, to_address, amount)

        mock_send_eth.assert_called_once_with(from_address, password, to_address, amount)


    @patch('core.ethereum.financial_manager.send_erc20')
    def test_execute_erc20_transaction(self, mock_send_erc20):
        from_address = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        to_address = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        password = "password123"
        amount = 10.0
        token_address = "0x6B175474E89094C44Da98b954EedeAC495271d0F" # DAI

        self.financial_manager.execute_transaction(from_address, password, to_address, amount, token_address)

        mock_send_erc20.assert_called_once_with(from_address, password, to_address, token_address, amount)

if __name__ == '__main__':
    unittest.main()