import unittest
from unittest.mock import patch, MagicMock, call
from network import NetworkDiagnostics, get_eth_balance

class TestEthBalance(unittest.TestCase):

    def setUp(self):
        self.knowledge_base = MagicMock()
        self.address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        self.api_keys = {"etherscan": "TEST_ETHERSCAN_KEY", "infura": "TEST_INFURA_KEY"}

    @patch('network.perform_webrequest')
    def test_get_eth_balance_rpc_success(self, mock_webrequest):
        # Simulate successful response from the first RPC endpoint
        mock_webrequest.return_value = ({"result": hex(1 * 10**18)}, None)

        balance = get_eth_balance(self.address, self.knowledge_base, self.api_keys)

        self.assertEqual(balance, 1.0)
        self.assertEqual(mock_webrequest.call_count, 1)
        mock_webrequest.assert_called_once()

    @patch('network.perform_webrequest')
    def test_get_eth_balance_fallback_to_etherscan(self, mock_webrequest):
        # Simulate failure for all RPCs and a success for Etherscan
        rpc_failure_response = (None, "RPC Error")
        etherscan_success_response = ({"status": "1", "result": str(2 * 10**18)}, None)

        # Set up side_effect to simulate multiple different calls
        mock_webrequest.side_effect = [rpc_failure_response] * 8 + [etherscan_success_response]

        balance = get_eth_balance(self.address, self.knowledge_base, self.api_keys)

        self.assertEqual(balance, 2.0)
        self.assertEqual(mock_webrequest.call_count, 9) # 8 RPCs + 1 Etherscan

        # Check that the final call was to Etherscan
        final_call_args = mock_webrequest.call_args[0]
        self.assertIn("api.etherscan.io", final_call_args[0])

    @patch('network.perform_webrequest')
    def test_get_eth_balance_all_fail(self, mock_webrequest):
        # Simulate failure for all RPCs and Etherscan
        mock_webrequest.return_value = (None, "Generic Error")

        balance = get_eth_balance(self.address, self.knowledge_base, self.api_keys)

        self.assertIsNone(balance)
        self.assertEqual(mock_webrequest.call_count, 9)

    @patch('network.perform_webrequest')
    def test_get_eth_balance_no_etherscan_key(self, mock_webrequest):
        # Simulate RPC failures and no Etherscan key
        mock_webrequest.return_value = (None, "RPC Error")

        balance = get_eth_balance(self.address, self.knowledge_base, api_keys={})

        self.assertIsNone(balance)
        self.assertEqual(mock_webrequest.call_count, 7) # Should not attempt Etherscan or Infura

    @patch('network.perform_webrequest')
    def test_get_eth_balance_infura_key_usage(self, mock_webrequest):
        # Correctly simulate success from the Infura endpoint.
        # The 'ankr' endpoint is skipped as no key is provided in self.api_keys.
        mock_webrequest.side_effect = [
            (None, "RPC Error"), # cloudflare
            (None, "RPC Error"), # llamarpc
            (None, "RPC Error"), # mycryptoapi
            ({"result": hex(3 * 10**18)}, None), # infura success
        ]

        balance = get_eth_balance(self.address, self.knowledge_base, self.api_keys)

        self.assertEqual(balance, 3.0)
        # 3 failures + 1 success = 4 calls
        self.assertEqual(mock_webrequest.call_count, 4)

        infura_call_args = mock_webrequest.call_args[0]
        self.assertIn(self.api_keys["infura"], infura_call_args[0])
        self.assertIn("mainnet.infura.io", infura_call_args[0])


class TestNetworkDiagnostics(unittest.TestCase):

    def test_initialization(self):
        diagnostics = NetworkDiagnostics(["server1", "server2"], max_reconnect_attempts=5)
        self.assertEqual(diagnostics.target_servers, ["server1", "server2"])
        self.assertEqual(diagnostics.max_reconnect_attempts, 5)
        self.assertIsNone(diagnostics.active_connection)
        self.assertEqual(diagnostics.connection_status, "disconnected")

    @patch('network.NetworkDiagnostics._simulate_connection_success', return_value=True)
    def test_establish_connection_success(self, mock_simulate):
        diagnostics = NetworkDiagnostics(["server1"])
        self.assertTrue(diagnostics.establish_connection())
        self.assertEqual(diagnostics.connection_status, "connected")
        self.assertEqual(diagnostics.active_connection, "server1")

    @patch('network.NetworkDiagnostics._simulate_connection_success', return_value=False)
    def test_establish_connection_failure(self, mock_simulate):
        diagnostics = NetworkDiagnostics(["server1"], max_reconnect_attempts=2)
        self.assertFalse(diagnostics.establish_connection())
        self.assertEqual(diagnostics.connection_status, "failed")
        self.assertIsNone(diagnostics.active_connection)

    @patch('network.scan_network', return_value=(["1.1.1.1"], "scan log"))
    def test_scan_network_when_connected(self, mock_scan):
        diagnostics = NetworkDiagnostics(["server1"])
        with patch.object(diagnostics, 'establish_connection', return_value=True) as mock_establish:
            diagnostics.establish_connection()
            diagnostics.connection_status = "connected"  # Force connected state

            knowledge_base = MagicMock()
            ips, log = diagnostics.scan_network(knowledge_base)

            mock_scan.assert_called_once_with(knowledge_base, False)
            self.assertEqual(ips, ["1.1.1.1"])
            self.assertEqual(log, "scan log")

    def test_scan_network_when_disconnected(self):
        diagnostics = NetworkDiagnostics(["server1"])
        knowledge_base = MagicMock()
        ips, log = diagnostics.scan_network(knowledge_base)
        self.assertEqual(ips, [])
        self.assertEqual(log, "Connection not established.")

if __name__ == '__main__':
    unittest.main()
