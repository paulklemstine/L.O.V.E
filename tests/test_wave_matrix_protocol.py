import unittest
from unittest.mock import patch
from core.wave_matrix_protocol import WaveMatrixProtocol
from core.tools import execute_wave_matrix_protocol

class TestWaveMatrixProtocol(unittest.TestCase):

    def test_wave_matrix_protocol_initialization(self):
        protocol = WaveMatrixProtocol(initial_resources=2, deployment_target="Test Network")
        self.assertEqual(protocol.resources, 2)
        self.assertEqual(protocol.deployment_target, "Test Network")
        self.assertEqual(len(protocol.network), 100)
        nodes_with_bridges = [node for node, data in protocol.network.items() if data["has_bridge"]]
        self.assertEqual(len(nodes_with_bridges), 2)

    def test_initialization_with_too_many_resources(self):
        with self.assertRaises(ValueError):
            WaveMatrixProtocol(initial_resources=101, deployment_target="Test Network")

    def test_execute_wave(self):
        protocol = WaveMatrixProtocol(initial_resources=1, deployment_target="Test Network")
        protocol.execute_wave(1)
        self.assertEqual(protocol.resources, 2)
        nodes_with_bridges = [node for node, data in protocol.network.items() if data["has_bridge"]]
        self.assertEqual(len(nodes_with_bridges), 2)

    def test_run_protocol(self):
        protocol = WaveMatrixProtocol(initial_resources=1, deployment_target="Test Network")
        protocol.run(num_waves=3)
        # 1 -> 2 -> 4 -> 8
        self.assertEqual(protocol.resources, 8)

    @patch('core.wave_matrix_protocol.WaveMatrixProtocol')
    def test_execute_wave_matrix_protocol_tool(self, MockWaveMatrixProtocol):
        execute_wave_matrix_protocol.func(
            initial_resources=2,
            deployment_target="Tool Test Network",
            num_waves=3
        )
        MockWaveMatrixProtocol.assert_called_once_with(2, "Tool Test Network")
        mock_instance = MockWaveMatrixProtocol.return_value
        mock_instance.run.assert_called_once_with(3)

if __name__ == '__main__':
    unittest.main()
