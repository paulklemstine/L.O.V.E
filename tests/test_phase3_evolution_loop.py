import unittest
from unittest import mock
import os
import json
from core.agents.metacognition_agent import MetacognitionAgent

class TestPhase3EvolutionLoop(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.log_dir = "logs"
        self.log_file = os.path.join(self.log_dir, "events.log")

        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Clean up old log files before each test
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists("core/tools_updated.py"):
            os.remove("core/tools_updated.py")

    def _seed_log_file_with_failure(self):
        """Helper to create a log file with a simulated tool failure."""
        log_entry = {
            "timestamp": "2023-10-27T10:00:00Z",
            "event_type": "tool_failure",
            "data": {
                "tool_name": "web_search",
                "error": "API limit exceeded"
            }
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def test_full_evolution_cycle(self):
        """
        Test the entire metacognitive loop from log analysis to PR submission.
        """
        # 1. Seed the log file with a failure event to trigger the loop
        self._seed_log_file_with_failure()

        # 2. Initialize and run the MetacognitionAgent
        meta_agent = MetacognitionAgent()

        # Mock the analyze_logs method to return a specific insight
        meta_agent.analyst.analyze_logs = mock.MagicMock(return_value="Insight: The web_search tool is inefficient.")

        # Mock the benchmarker to avoid running a real experiment
        meta_agent.benchmarker.run_experiment = mock.MagicMock(return_value=True)

        # Provide a mock log entry for the agent to analyze
        mock_logs = [{"event_type": "tool_failure", "data": {"tool_name": "web_search"}}]
        meta_agent.run_evolution_cycle(logs=mock_logs)

        # 3. Assertions to verify the process
        # Check that analyze_logs was called with the mock logs
        meta_agent.analyst.analyze_logs.assert_called_once_with(mock_logs)

        # Check that the placeholder file for the new code was created as a side effect
        self.assertTrue(os.path.exists("core/tools_updated.py"), "The new tool code file was not created.")

if __name__ == '__main__':
    unittest.main()