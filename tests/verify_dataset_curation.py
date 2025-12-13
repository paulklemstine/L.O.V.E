
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.monitoring import MonitoringManager

class TestDatasetCuration(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_love_state = {
            'monitoring': {
                'cpu_usage': [],
                'mem_usage': [],
                'anomalies': [],
                'task_history': []
            },
            'love_tasks': {}
        }
        self.mock_console = MagicMock()
        self.monitor = MonitoringManager(self.mock_love_state, self.mock_console)

    @patch('langsmith.Client')
    def test_scan_traces_adds_to_dataset(self, MockClient):
        # Setup Environment
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # Mock Client Instance
        mock_client_instance = MockClient.return_value
        
        # Mock Dataset (create or read)
        mock_dataset = MagicMock()
        mock_dataset.id = "mock-dataset-id"
        mock_client_instance.read_dataset.return_value = mock_dataset
        
        # Mock List Runs (Return one high-score run and one low-score run)
        mock_run_good = MagicMock()
        mock_run_good.id = "good-run-id"
        mock_run_good.inputs = {"prompt": "foo"}
        mock_run_good.outputs = {"response": "bar"}
        # Note: In reality, filter happens server-side, but client.list_runs returns the iterator.
        # We are mocking list_runs to return specific runs.
        # We assume _scan_traces passes the correct filter string, but we can't easily mock the server-side filtering logic here.
        # So we just provide the run that WOULD be returned by the filter.
        
        mock_client_instance.list_runs.return_value = [mock_run_good]
        
        # Execute
        self.monitor._scan_traces()
        
        # Verify read_dataset called (or create)
        mock_client_instance.read_dataset.assert_called_with(dataset_name="gold-standard-behaviors")
        
        # Verify create_example called for good run
        mock_client_instance.create_example.assert_called_with(
            inputs={"prompt": "foo"},
            outputs={"response": "bar"},
            dataset_id="mock-dataset-id"
        )
        
        # Verify run is marked processed
        self.assertIn("good-run-id", self.monitor._processed_run_ids)
        
        # Execute again (should not duplicate)
        self.monitor._scan_traces()
        mock_client_instance.create_example.assert_called_once() # Count remains 1

if __name__ == '__main__':
    unittest.main()
