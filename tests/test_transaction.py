import unittest
from unittest.mock import patch, MagicMock
from transaction import send_eth, send_erc20

class TestTransaction(unittest.TestCase):

    @patch('transaction.getpass')
    @patch('transaction.get_private_key')
    @patch('transaction.web3')
    def test_send_eth_success(self, mock_web3, mock_get_private_key, mock_getpass):
        # Arrange
        mock_get_private_key.return_value = '0x' + 'a' * 64
        mock_web3.eth.get_transaction_count.return_value = 0
        mock_web3.eth.gas_price = 50000000000
        mock_web3.to_wei.return_value = 1000000000000000000
        mock_web3.eth.account.sign_transaction.return_value = MagicMock(rawTransaction=b'raw_tx')
        mock_web3.eth.send_raw_transaction.return_value = b'tx_hash'
        mock_web3.to_hex.return_value = '0x' + 'b' * 64

        # Act
        with patch('builtins.print') as mock_print:
            result = send_eth('0x' + 'c' * 40, 1)

        # Assert
        self.assertEqual(result, '0x' + 'b' * 64)
        mock_print.assert_any_call("Transaction sent! Tx Hash: 0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    @patch('transaction.getpass')
    @patch('transaction.get_private_key')
    @patch('transaction.web3')
    def test_send_erc20_success(self, mock_web3, mock_get_private_key, mock_getpass):
        # Arrange
        mock_get_private_key.return_value = '0x' + 'a' * 64

        mock_contract = MagicMock()
        mock_contract.functions.decimals.return_value.call.return_value = 18
        mock_contract.functions.transfer.return_value.build_transaction.return_value = {}
        mock_web3.eth.contract.return_value = mock_contract
        mock_web3.eth.get_transaction_count.return_value = 0
        mock_web3.eth.gas_price = 50000000000
        mock_web3.eth.account.sign_transaction.return_value = MagicMock(rawTransaction=b'raw_tx')
        mock_web3.eth.send_raw_transaction.return_value = b'tx_hash'
        mock_web3.to_hex.return_value = '0x' + 'b' * 64

        # Act
        with patch('builtins.print') as mock_print:
            result = send_erc20('0x' + 'c' * 40, '0x' + 'd' * 40, 1)

        # Assert
        self.assertEqual(result, '0x' + 'b' * 64)
        mock_print.assert_any_call("Transaction sent! Tx Hash: 0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

if __name__ == '__main__':
    unittest.main()