
import unittest
from unittest.mock import patch, MagicMock

from ethereum_staking import stake_ethereum

class TestEthereumStaking(unittest.TestCase):

    @patch('ethereum_staking.YieldXYZClient')
    def test_stake_ethereum_success(self, MockYieldXYZClient):
        """
        Tests that stake_ethereum successfully calls the Yield.xyz client with the correct parameters.
        """
        # Arrange: Set up the mock client and its return value
        mock_client_instance = MockYieldXYZClient.return_value
        mock_transaction = {
            "status": "success",
            "unsigned_transaction": "0x12345..."
        }
        mock_client_instance.enter_yield.return_value = mock_transaction

        # Act: Call the function with test data
        api_key = "test_api_key"
        yield_id = "ethereum-lido-staking"
        address = "0xAbCdEf1234567890"
        amount = 0.5
        result = stake_ethereum(api_key, yield_id, address, amount)

        # Assert: Check that the client was called correctly and the result is as expected
        amount_in_wei = str(int(amount * 1e18))
        mock_client_instance.enter_yield.assert_called_once_with(
            yield_id=yield_id,
            address=address,
            amount=amount_in_wei
        )
        self.assertEqual(result, mock_transaction)

    @patch('ethereum_staking.YieldXYZClient')
    def test_stake_ethereum_api_error(self, MockYieldXYZClient):
        """
        Tests that stake_ethereum handles exceptions from the Yield.xyz client gracefully.
        """
        # Arrange: Configure the mock client to raise an exception
        mock_client_instance = MockYieldXYZClient.return_value
        error_message = "API limit reached"
        mock_client_instance.enter_yield.side_effect = Exception(error_message)

        # Act: Call the function and expect an error response
        api_key = "test_api_key"
        yield_id = "ethereum-lido-staking"
        address = "0xAbCdEf1234567890"
        amount = 0.5
        result = stake_ethereum(api_key, yield_id, address, amount)

        # Assert: Verify that the function returns a structured error message
        self.assertIn("status", result)
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)
        self.assertEqual(result["message"], error_message)

if __name__ == '__main__':
    unittest.main()
