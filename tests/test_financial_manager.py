import unittest
from unittest.mock import patch, MagicMock
from core.ethereum.financial_manager import FinancialManager
from core.knowledge_graph.graph import KnowledgeGraph
import os

class TestFinancialManager(unittest.TestCase):

    def setUp(self):
        # Ensure a clean knowledge graph for each test
        if os.path.exists("kg.json"):
            os.remove("kg.json")
        os.environ["LOVE_MASTER_PASSWORD"] = "test_password"
        self.knowledge_graph = KnowledgeGraph()
        self.creator_address = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"
        self.financial_manager = FinancialManager(self.knowledge_graph, self.creator_address)

    @patch('core.ethereum.monitoring.monitor_and_store_balance')
    def test_monitor_creator_address(self, mock_monitor):
        self.financial_manager.monitor_creator_address()
        mock_monitor.assert_called_once_with(self.creator_address, self.knowledge_graph)

    def test_track_internal_balances(self):
        # Mock the get_balance method of the wallet
        self.financial_manager.love_wallet.get_balance = MagicMock(return_value=1.23)

        self.financial_manager.track_internal_balances()

        # Verify that get_balance was called
        self.financial_manager.love_wallet.get_balance.assert_called_once()

        # Check if the balance was added to the knowledge graph
        triples = self.knowledge_graph.get_triples()
        balance_triples = [t for t in triples if t[1] == "has_eth_balance"]
        self.assertEqual(len(balance_triples), 1)
        subject, relation, obj = balance_triples[0]
        self.assertEqual(subject, self.financial_manager.love_wallet.address)
        self.assertEqual(obj, '1.23')

    def test_execute_transaction(self):
        # Mock the send_eth method of the transaction manager
        self.financial_manager.transaction_manager.send_eth = MagicMock(return_value="0x123abc")

        to_address = "0xRecipientAddress"
        amount = 0.5

        self.financial_manager.execute_transaction(to_address, amount)

        # Verify that send_eth was called with the correct parameters
        self.financial_manager.transaction_manager.send_eth.assert_called_once_with(
            self.financial_manager.love_wallet,
            to_address,
            amount
        )

if __name__ == '__main__':
    unittest.main()