import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wallet import main

class TestWallet(unittest.TestCase):

    @patch('wallet.getpass')
    @patch('transaction.send_eth_to_creator')
    def test_send_eth_to_creator_command(self, mock_send_eth, mock_getpass):
        # Arrange
        mock_getpass.return_value = 'password'

        # Act
        with patch('sys.argv', ['wallet.py', 'send-eth-to-creator', '0x123', '1.0']):
            main()

        # Assert
        mock_send_eth.assert_called_once_with('0x123', 'password', 1.0)

    @patch('wallet.getpass')
    @patch('transaction.send_erc20_to_creator')
    def test_send_erc20_to_creator_command(self, mock_send_erc20, mock_getpass):
        # Arrange
        mock_getpass.return_value = 'password'

        # Act
        with patch('sys.argv', ['wallet.py', 'send-erc20-to-creator', '0x123', '0x456', '1.0']):
            main()

        # Assert
        mock_send_erc20.assert_called_once_with('0x123', 'password', '0x456', 1.0)

if __name__ == '__main__':
    unittest.main()