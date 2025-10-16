import unittest
import sys
import os
import asyncio
from unittest.mock import patch

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.orchestrator import Orchestrator

class TestPhase2Integration(unittest.IsolatedAsyncioTestCase):
    """
    An integration test to ensure all components of Phase 2 (Planning, Tools,
    and Execution) work together as a system.
    """

    def setUp(self):
        """Set up the test environment."""
        print("\n--- Setting up TestPhase2Integration ---")
        self.orchestrator = Orchestrator()

    async def test_full_execution_flow(self):
        """
        Tests the entire workflow from receiving a high-level goal to
        executing a multi-step plan and producing a final result.
        """
        # Define the high-level goal for the test
        goal = "Summarize the latest advancements in AI"

        print(f"\n--- Running test_full_execution_flow with goal: '{goal}' ---")

        # Execute the goal using the orchestrator
        result = await self.orchestrator.execute_goal(goal)

        # --- Assertions ---

        # 1. Check that the overall status is 'Success'
        self.assertEqual(result['status'], 'Success', "The plan should execute successfully.")

        # 2. Check the final result
        self.assertIn("Article 1 Content", result['final_result'],
                      "The final result should be the content of the article from the previous step.")

        # 3. Inspect the plan state for correctness
        plan_state = result.get('plan_state', [])
        self.assertEqual(len(plan_state), 5, "The plan should have exactly 5 steps.")

        # 4. Verify that all steps were successful
        for i, step_info in enumerate(plan_state):
            self.assertEqual(step_info['status'], 'success', f"Step {i+1} should be marked as 'success'.")
            self.assertIsNotNone(step_info['result'], f"Step {i+1} should have a result.")

        print("\n--- Test 'test_full_execution_flow' PASSED ---")

    @patch('core.planning.mock_llm_call')
    async def test_error_handling_for_unknown_tool(self, mock_llm_call_func):
        """
        Tests how the system handles a task that requires a tool that is not registered.
        """
        # A goal that will generate a plan with a step we can't execute
        goal = "Launch a rocket to Mars"

        # To make this test predictable, we'll manually set a plan that
        # contains a step with an unknown tool.
        mock_plan_json = """
        [
            {"step": 1, "task": "Design the rocket."},
            {"step": 2, "task": "Launch the rocket using the 'launch_rocket' tool.", "tool": "launch_rocket", "args": {}}
        ]
        """
        mock_llm_call_func.return_value = mock_plan_json


        print(f"\n--- Running test_error_handling_for_unknown_tool with goal: '{goal}' ---")

        # Execute the goal
        result = await self.orchestrator.execute_goal(goal)

        # --- Assertions ---

        # 1. The overall status should be 'Failed'
        self.assertEqual(result['status'], 'Failed', "The plan should fail due to the unknown tool.")

        # 2. The reason for failure should be accurate
        self.assertIn('A step failed', result['reason'])
        self.assertIn("Tool 'launch_rocket' is not registered", result['reason'])

        # 3. Check the state of the failed step
        plan_state = result.get('plan_state', [])
        failed_step = plan_state[1] # The second step should have failed
        self.assertEqual(failed_step['status'], 'failed', "The second step should be marked 'failed'.")
        self.assertIn("Error: Tool 'launch_rocket' is not registered.", failed_step['result'],
                      "The result of the failed step should contain the correct error message.")

        print("\n--- Test 'test_error_handling_for_unknown_tool' PASSED ---")

if __name__ == '__main__':
    unittest.main()