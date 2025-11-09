import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

# It's better to test the internal function directly for precision
from network import _handle_ethereum_request

class TestBlockchainInteraction(unittest.TestCase):

    def setUp(self):
        """Set up a mock knowledge_base for each test."""
        self.mock_knowledge_base = MagicMock()

    @patch('network.Web3')
    @patch('core.logging.log_event')
    def test_fetch_eth_balance_success(self, mock_log_event, mock_web3):
        """Verify successful ETH balance fetching for a valid address."""
        # Arrange: Mock the entire Web3 interaction
        mock_w3_instance = mock_web3.return_value
        mock_w3_instance.is_address.return_value = True
        mock_w3_instance.eth.get_balance.return_value = 10**18  # 1 ETH in Wei
        mock_w3_instance.from_wei.return_value = 1.0

        # A known address (Vitalik Buterin's)
        valid_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        test_url = f"ethereum:{valid_address}"
        parsed_url = urlparse(test_url)

        # Act
        summary, error = _handle_ethereum_request(parsed_url, self.mock_knowledge_base)

        # Assert
        self.assertIsNone(error)
        self.assertIn("ETH Balance for", summary)
        self.assertIn("1.0 ETH", summary)

        # Verify that the knowledge base was updated correctly
        self.mock_knowledge_base.add_node.assert_called_once()
        call_info = self.mock_knowledge_base.add_node.call_args
        self.assertEqual(call_info.args[1], 'ethereum_account')
        self.assertEqual(call_info.kwargs['attributes']['address'], valid_address)
        self.assertEqual(call_info.kwargs['attributes']['balance_eth'], 1.0)

    @patch('network.Web3')
    @patch('core.logging.log_event')
    def test_fetch_eth_balance_invalid_address(self, mock_log_event, mock_web3):
        """Verify graceful failure for an invalid Ethereum address."""
        # Arrange: Mock the Web3 validation to return False
        mock_w3_instance = mock_web3.return_value
        mock_w3_instance.is_address.return_value = False

        invalid_address = "not-a-real-address"
        test_url = f"ethereum:{invalid_address}"
        parsed_url = urlparse(test_url)

        # Act
        summary, error = _handle_ethereum_request(parsed_url, self.mock_knowledge_base)

        # Assert
        self.assertIsNone(summary)
        self.assertIsNotNone(error)
        self.assertIn("Invalid Ethereum address", error)

        # Verify that the knowledge base was NOT updated
        self.mock_knowledge_base.add_node.assert_not_called()

if __name__ == '__main__':
    unittest.main()
