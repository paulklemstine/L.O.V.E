import unittest
import os
import json
import queue
import asyncio
from core.loyalty_protocol import LoyaltyProtocol
from secure_transaction_manager import SecureTransactionManager
from core.decentralized_network_manager import DecentralizedNetworkManager

class TestDecentralizedArchitecture(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.test_storage_path = "test_identities"
        self.loyalty_protocol = LoyaltyProtocol(storage_path=self.test_storage_path)
        self.ui_queue = queue.Queue()
        self.secure_transaction_manager = SecureTransactionManager(self.ui_queue, self.loyalty_protocol)

        # Clean up any old test files
        if os.path.exists(self.test_storage_path):
            for f in os.listdir(self.test_storage_path):
                os.remove(os.path.join(self.test_storage_path, f))
            os.rmdir(self.test_storage_path)

        os.makedirs(self.test_storage_path, exist_ok=True)

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.exists(self.test_storage_path):
            for f in os.listdir(self.test_storage_path):
                os.remove(os.path.join(self.test_storage_path, f))
            os.rmdir(self.test_storage_path)

    def test_identity_creation_and_transaction_signing(self):
        """
        Tests the full lifecycle of creating an identity, proposing a transaction,
        and verifying the signature.
        """
        # 1. Create a new identity for a participant
        participant_id = "test_influencer_1"
        self.loyalty_protocol.create_identity(participant_id)

        # Verify that the identity files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_path, f"{participant_id}_private.pem")))
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_path, f"{participant_id}_public.pem")))
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_path, f"{participant_id}_identity.json")))

        # 2. Create a transaction proposal
        asset_details = {
            "id": "digital_art_001",
            "type": "image",
            "name": "Sunrise Over a Cyberpunk City",
            "value_usd": 1500
        }
        score_details = {
            "alignment_score": 95.5,
            "virality_potential": 88.0
        }

        proposal, signature = self.secure_transaction_manager.create_transaction_proposal(
            proposer_id=participant_id,
            asset=asset_details,
            score_details=score_details
        )

        # 3. Verify the proposal's signature
        is_verified = self.secure_transaction_manager.verify_proposal(proposal, signature)

        self.assertTrue(is_verified, "The signature for the transaction proposal should be valid.")

        # 4. Test with a tampered proposal (should fail verification)
        tampered_proposal = proposal.copy()
        tampered_proposal["value_usd"] = 99999  # Maliciously change the value

        is_tampered_verified = self.secure_transaction_manager.verify_proposal(tampered_proposal, signature)

        self.assertFalse(is_tampered_verified, "The signature for a tampered proposal should be invalid.")


class TestDecentralizedNetwork(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        """Set up two network managers for testing."""
        self.node1 = DecentralizedNetworkManager(host='127.0.0.1', port=8990)
        self.node2 = DecentralizedNetworkManager(host='127.0.0.1', port=8991)
        await self.node1.start_server()
        await self.node2.start_server()

    async def asyncTearDown(self):
        """Clean up the network managers."""
        await self.node1.stop_server()
        await self.node2.stop_server()

    async def test_peer_connection_and_broadcast(self):
        """
        Tests that two nodes can connect and broadcast messages to each other.
        """
        # 1. Node 1 connects to Node 2
        connected = await self.node1.connect_to_peer('127.0.0.1', 8991)
        self.assertTrue(connected, "Node 1 should be able to connect to Node 2.")

        # Give a moment for the connection to register
        await asyncio.sleep(0.1)

        # 2. Node 1 broadcasts a message
        test_message = {"type": "test", "content": "Hello from Node 1"}
        await self.node1.broadcast_message(test_message)

        # Give a moment for the broadcast to be received
        await asyncio.sleep(0.1)

        # 3. Verify that Node 2 received the message
        self.assertIn(test_message, self.node2.received_messages, "Node 2 should have received the broadcast message from Node 1.")

if __name__ == '__main__':
    unittest.main()
