import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.secure_transaction_manager import SecureTransactionManager

class TestSecureTransactionManager(unittest.TestCase):
    def test_create_investment_proposal(self):
        # Create a mock for the websocket manager
        websocket_manager = MagicMock()
        websocket_manager.broadcast = AsyncMock()

        # Create an instance of the SecureTransactionManager
        transaction_manager = SecureTransactionManager(websocket_manager)

        # Define the proposal details
        proposal_details = {
            'name': 'Test Proposal',
            'symbol': 'TEST',
            'trend_score': 1.0,
            'price_change_24h': 0.0,
            'reason': 'Test Reason'
        }

        # Create the investment proposal
        asyncio.run(transaction_manager.create_investment_proposal(proposal_details))

        # Assert that the broadcast method was called with the correct arguments
        websocket_manager.broadcast.assert_called_once()
        call_args = websocket_manager.broadcast.call_args[0][0]
        self.assertEqual(call_args['type'], 'investment_proposal')
        self.assertEqual(call_args['details'], proposal_details)

if __name__ == '__main__':
    unittest.main()
