import unittest
import sys
import os
import asyncio
from unittest.mock import patch

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.orchestrator import Orchestrator
from core.graph_manager import GraphDataManager
from core.memory.memory_manager import MemoryManager
from unittest.mock import MagicMock

class TestPhase2Integration(unittest.IsolatedAsyncioTestCase):
    """
    An integration test to ensure all components of Phase 2 (Planning, Tools,
    and Execution) work together as a system.
    """

    def setUp(self):
        """Set up the test environment."""
        print("\n--- Setting up TestPhase2Integration ---")
        mock_graph_manager = MagicMock(spec=GraphDataManager)
        mock_memory_manager = MemoryManager(mock_graph_manager)
        self.orchestrator = Orchestrator(mock_memory_manager)

    # async def test_full_execution_flow(self):
    #     """
    #     Tests the entire workflow from receiving a high-level goal to
    #     executing a multi-step plan and producing a final result.
    #     """
    #     # Define the high-level goal for the test
    #     goal = "Summarize the latest advancements in AI"

    #     print(f"\n--- Running test_full_execution_flow with goal: '{goal}' ---")

    #     # Execute the goal using the orchestrator
    #     result = await self.orchestrator.execute_goal(goal)

    #     # --- Assertions ---

    #     # 1. Check that the overall status is 'Success'
    #     self.assertEqual(result['status'], 'Success', "The plan should execute successfully.")

    #     # 2. Check the final result
    #     self.assertIn("Article 1 Content", result['final_result'],
    #                   "The final result should be the content of the article from the previous step.")

    #     # 3. Inspect the plan state for correctness
    #     plan_state = result.get('plan_state', [])
    #     self.assertEqual(len(plan_state), 5, "The plan should have exactly 5 steps.")

    #     # 4. Verify that all steps were successful
    #     for i, step_info in enumerate(plan_state):
    #         self.assertEqual(step_info['status'], 'success', f"Step {i+1} should be marked as 'success'.")
    #         self.assertIsNotNone(step_info['result'], f"Step {i+1} should have a result.")

    #     print("\n--- Test 'test_full_execution_flow' PASSED ---")

if __name__ == '__main__':
    unittest.main()