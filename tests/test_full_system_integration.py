import unittest
import json
from unittest.mock import patch, MagicMock

# Add necessary imports from the 'core' directory
from core.agents.orchestrator import Orchestrator
from core.metacognition import HypothesisFormatter, ExperimentPlanner
from core.structured_logger import StructuredEventLogger
from core.agents.analyst_agent import AnalystAgent

class TestFullSystemIntegration(unittest.TestCase):
    """
    An integration test that validates the harmonious operation of all three phases
    of the agent's architecture:
    1.  Cognitive Architecture (Initialization)
    2.  Action & Planning Engine (Goal Execution)
    3.  Metacognitive Evolution Loop (Self-Improvement)
    """

    def test_phase_1_cognitive_architecture_initialization(self):
        """
        Validates that the Orchestrator and its core components are
        initialized correctly, confirming the success criterion of Step 1.1.
        """
        print("\n--- Running Test for Phase 1: Cognitive Architecture ---")

        # Instantiate the Orchestrator
        orchestrator = Orchestrator()

        # Verify that all components are initialized
        self.assertIsNotNone(orchestrator, "Orchestrator should be initialized.")
        self.assertIsNotNone(orchestrator.planner, "Planner should be initialized.")
        self.assertIsNotNone(orchestrator.tool_registry, "Tool Registry should be initialized.")
        self.assertIsNotNone(orchestrator.executor, "Executor should be initialized.")
        self.assertIsNotNone(orchestrator.execution_engine, "Execution Engine should be initialized.")

        # Verify that the tool registry contains the expected tools
        self.assertIsNotNone(orchestrator.tool_registry.get_tool("web_search"))
        self.assertIsNotNone(orchestrator.tool_registry.get_tool("read_file"))

        print("--- Phase 1 Test Passed: Cognitive Architecture is sound. ---")

    @patch('core.agents.orchestrator.read_file')
    @patch('core.agents.orchestrator.web_search')
    @patch('core.planning.mock_llm_call')
    def test_phase_2_action_and_planning_engine(self, mock_llm_call_func, mock_web_search, mock_read_file):
        """
        Validates that the agent can take a high-level goal, decompose it
        into a plan, and execute it, confirming the success criteria of
        Steps 2.1, 2.2, and 2.3.
        """
        print("\n--- Running Test for Phase 2: Action & Planning Engine ---")

        # Configure the mock LLM to return a specific plan for our test goal
        test_goal = "Summarize the latest advancements in AI"
        mock_plan = [
            {"step": 1, "task": "Identify key research sources."},
            {"step": 2, "task": "Execute web searches using the 'web_search' tool."},
            {"step": 3, "task": "Read the content of the top article using the 'read_file' tool."},
            {"step": 4, "task": "Produce a final summary report."}
        ]
        mock_llm_call_func.return_value = json.dumps(mock_plan)

        # Configure mock tool behaviors
        mock_article_content = "The latest advancements in AI are groundbreaking."
        mock_web_search.return_value = "[Article 1: AI Today, Article 2: Future of AI]"
        mock_read_file.return_value = mock_article_content

        # Instantiate the Orchestrator
        orchestrator = Orchestrator()

        # Execute the goal
        result = orchestrator.execute_goal(test_goal)

        # --- Assertions ---
        # 1. Check if the overall execution was successful
        self.assertEqual(result.get('status'), 'Success', "Execution should be successful.")

        # 2. Check if the final result is as expected
        self.assertEqual(result.get('final_result'), mock_article_content, "The final summary is incorrect.")

        # 3. Check the state of the plan to ensure all steps were successful
        plan_state = result.get('plan_state', [])
        self.assertTrue(all(s['status'] == 'success' for s in plan_state), "All steps in the plan should have succeeded.")

        # 4. Verify that the mock LLM and tools were called correctly
        mock_llm_call_func.assert_called_once()
        mock_web_search.assert_called()
        mock_read_file.assert_called_once()

        print("--- Phase 2 Test Passed: Action & Planning Engine is functional. ---")

    def test_phase_3_metacognitive_evolution_loop(self):
        """
        Validates the self-improvement loop by simulating the generation of
        an insight, hypothesis, and experiment plan, confirming the success
        criteria of Steps 3.1 and 3.2.
        """
        print("\n--- Running Test for Phase 3: Metacognitive Evolution Loop ---")

        # 1. Simulate performance logs (as the input for the AnalystAgent)
        mock_logs = [
            {"event": "tool_start", "tool_name": "web_search", "token_usage": 1500},
            {"event": "tool_success", "tool_name": "web_search", "result": "Full HTML page"},
            {"event": "tool_start", "tool_name": "web_search", "token_usage": 600},
            {"event": "tool_success", "tool_name": "web_search", "result": "Another full HTML page"}
        ]

        # 2. Use the AnalystAgent to generate an insight
        insight = AnalystAgent.analyze_logs(mock_logs)
        self.assertIn("Insight: The web_search tool is inefficient", insight, "Analyst Agent should identify inefficiency.")

        # 3. Use the HypothesisFormatter to create a testable hypothesis
        hypothesis = HypothesisFormatter.format_hypothesis(insight)
        self.assertTrue(hypothesis.startswith("IF we"), "Hypothesis should be correctly formatted.")
        self.assertIn("decrease by over 50%", hypothesis, "Hypothesis should predict a significant decrease.")

        # 4. Use the ExperimentPlanner to design a validation experiment
        experiment = ExperimentPlanner.design_experiment(hypothesis)
        self.assertEqual(experiment['metric'], 'token_usage_metric', "Experiment metric should be 'token_usage_metric'.")
        self.assertIn("control.token_usage * 0.5", experiment['success_condition'], "Success condition should be a 50% reduction.")

        print("--- Phase 3 Test Passed: Metacognitive Loop is functional. ---")

if __name__ == "__main__":
    unittest.main()