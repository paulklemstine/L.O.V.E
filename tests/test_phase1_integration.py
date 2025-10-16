import unittest
import asyncio
from unittest.mock import patch, MagicMock

from core.agents.orchestrator import Orchestrator

class TestPhase1Integration(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.orchestrator = Orchestrator()

    def test_orchestrator_initialization(self):
        """
        Tests if the Orchestrator and its key components are initialized correctly.
        """
        print("\n--- Running test_orchestrator_initialization ---")
        self.assertIsNotNone(self.orchestrator.planner)
        self.assertIsNotNone(self.orchestrator.tool_registry)
        self.assertIsNotNone(self.orchestrator.executor)
        self.assertIsNotNone(self.orchestrator.execution_engine)
        # Check if default tools are registered
        self.assertIn("web_search", self.orchestrator.tool_registry.list_tools())
        self.assertIn("read_file", self.orchestrator.tool_registry.list_tools())

    @patch('core.planning.mock_llm_call')
    def test_simple_goal_execution(self, mock_llm_call_func):
        """
        Tests a simple, successful workflow where the orchestrator takes a goal,
        generates a plan, and executes it. This replaces the old agent registration test.
        """
        # --- Arrange ---
        goal = "Summarize the latest advancements in AI"
        print(f"\n--- Running test_simple_goal_execution with goal: '{goal}' ---")

        # Mock the LLM call within the planner to return a predictable plan
        mock_plan_json = """
        [
            {"step": 1, "task": "Execute web search for 'latest AI advancements'", "tool": "web_search", "args": {"query": "latest AI advancements"}},
            {"step": 2, "task": "Read the content of the first article", "tool": "read_file", "args": {"path": "/mnt/data/article1.txt"}},
            {"step": 3, "task": "Synthesize a final summary."}
        ]
        """
        mock_llm_call_func.return_value = mock_plan_json

        # Mock the executor to avoid actual tool execution
        # We need an async mock for the executor's async `execute` method
        async def mock_executor_execute(tool_name, tool_registry, **kwargs):
            if tool_name == "web_search":
                return '[{"title": "AI Advancements", "url": "/mnt/data/article1.txt"}]'
            if tool_name == "read_file":
                return "Article content about AI."
            return "Default mock result"

        self.orchestrator.executor.execute = MagicMock(side_effect=mock_executor_execute)

        # --- Act ---
        # The execute_goal method is async, so we need to run it in the event loop.
        result = asyncio.run(self.orchestrator.execute_goal(goal))

        # --- Assert ---
        self.assertIsNotNone(result)
        self.assertEqual(result.get('status'), 'Success')
        self.assertIn("summary", result.get('final_result', '').lower()) # More flexible check

        # Verify that the planner was called
        mock_llm_call_func.assert_called_once()
        # Verify that the executor was called for the tools in the plan
        self.assertEqual(self.orchestrator.executor.execute.call_count, 2)


if __name__ == '__main__':
    unittest.main()