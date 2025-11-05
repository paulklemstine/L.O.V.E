import unittest
from unittest.mock import patch, MagicMock
from network import NetworkDiagnostics

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
