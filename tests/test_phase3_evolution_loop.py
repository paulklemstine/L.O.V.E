import unittest
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
        meta_agent.run_evolution_cycle()

        # 3. Assertions to verify the process
        # For this test, we are checking that the simulated process runs to completion.
        # A more robust test would mock each component and check the calls and outputs.

        # Check that the placeholder file for the new code was created
        self.assertTrue(os.path.exists("core/tools_updated.py"), "The new tool code file was not created.")

if __name__ == '__main__':
    unittest.main()