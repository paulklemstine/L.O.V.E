import pytest
import unittest
from unittest.mock import patch, MagicMock
from ipfs import pin_to_ipfs_sync

class TestIPFSIntegration(unittest.TestCase):

    @patch('requests.post')
    def test_pin_to_ipfs_sync_optimization(self, mock_post):
        """
        Verify that pin_to_ipfs_sync makes only one call to /add and does NOT call /id.
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Hash': 'QmTestHash'}
        mock_post.return_value = mock_response

        # Call function
        cid = pin_to_ipfs_sync(b"test data", None)

        # Verify result
        self.assertEqual(cid, 'QmTestHash')

        # Verify only ONE call was made (the /add call)
        self.assertEqual(mock_post.call_count, 1)

        # Verify arguments
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:5002/api/v0/add")
        self.assertEqual(kwargs['timeout'], 2.0)

    @patch('requests.post')
    def test_pin_to_ipfs_sync_handles_connection_error(self, mock_post):
        """
        Verify that pin_to_ipfs_sync gracefully handles connection errors without the explicit check.
        """
        # Simulate connection error
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        # Call function
        cid = pin_to_ipfs_sync(b"test data", None)

        # Verify result is None (safe failure)
        self.assertIsNone(cid)

        # Verify only one call was attempted
        self.assertEqual(mock_post.call_count, 1)

if __name__ == '__main__':
    unittest.main()
