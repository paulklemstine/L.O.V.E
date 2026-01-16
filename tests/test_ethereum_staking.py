
import unittest
from unittest.mock import patch
from ethereum_staking import stake_ethereum

class TestEthereumStaking(unittest.TestCase):

    @patch('ethereum_staking.log_event')
    def test_stake_ethereum(self, mock_log_event):
        """
        Tests the stake_ethereum function to ensure it returns the correct
        details and logs the appropriate message.
        """
        amount = 5.0
        expected_result = {
            "status": "success",
            "platform": "ExampleYield",
            "asset": "Ethereum (ETH)",
            "amount_staked": amount,
            "confirmation_message": f"Successfully staked {amount} ETH. Your assets are now earning interest."
        }

        # Call the function
        result = stake_ethereum(amount)

        # Assert the result is as expected
        self.assertEqual(result, expected_result)

        # Assert that the log_event function was called with the correct message
        mock_log_event.assert_called_once_with(f"SIMULATION: Staking {amount} ETH on ExampleYield.", "INFO")

if __name__ == '__main__':
    unittest.main()
