
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
import asyncio
import json

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.polly import PollyOptimizer

class TestPollySafeguards(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Patch the Client to avoid real LangSmith calls
        self.langsmith_patcher = patch("core.polly.Client")
        self.MockClient = self.langsmith_patcher.start()
        
        # Patch prompt registry
        self.registry_patcher = patch("core.polly.get_prompt_registry")
        self.MockRegistry = self.registry_patcher.start()
        self.mock_registry_instance = MagicMock()
        self.MockRegistry.return_value = self.mock_registry_instance
        self.mock_registry_instance.get_prompt.return_value = "Original Prompt Content"

        # Mock golden dataset loading to avoid file dependency errors or empty dataset
        self.dataset_patcher = patch.object(PollyOptimizer, '_load_golden_dataset')
        self.mock_load_dataset = self.dataset_patcher.start()
        self.mock_load_dataset.return_value = [
            {"id": "test_1", "input": "Test Input 1", "criteria": "Criteria 1"},
            {"id": "test_2", "input": "Test Input 2", "criteria": "Criteria 2"}
        ]
        
    def tearDown(self):
        self.langsmith_patcher.stop()
        self.registry_patcher.stop()
        self.dataset_patcher.stop()

    @patch("core.polly.run_llm", new_callable=AsyncMock)
    async def test_recursion_limit(self, mock_run_llm):
        optimizer = PollyOptimizer()
        
        # Call with depth 2 (max is 2)
        result = await optimizer.optimize_prompt("test_key", recursion_depth=2)
        
        # Should return None and log warning (not checked here but result is None)
        self.assertIsNone(result)
        # Should NOT call LLM
        mock_run_llm.assert_not_called()

    @patch("core.polly.run_llm", new_callable=AsyncMock)
    async def test_evaluate_candidate_prompt(self, mock_run_llm):
        optimizer = PollyOptimizer()
        
        # Configure mock_run_llm side effects
        # 1. Task execution for Example 1
        # 2. Judge for Example 1
        # 3. Task execution for Example 2
        # 4. Judge for Example 2
        
        mock_run_llm.side_effect = [
            {"result": "Task Response 1"}, # Task 1
            {"result": "Score: 8"},       # Judge 1
            {"result": "Task Response 2"}, # Task 2
            {"result": "Score: 6"}        # Judge 2
        ]
        
        score = await optimizer.evaluate_candidate_prompt("New Prompt Candidate")
        
        # Expected average: (8 + 6) / 2 = 7.0
        self.assertEqual(score, 7.0)
        self.assertEqual(mock_run_llm.call_count, 4)

    @patch("core.polly.run_llm", new_callable=AsyncMock)
    async def test_optimize_prompt_promotion(self, mock_run_llm):
        """Test that a better prompt is promoted."""
        optimizer = PollyOptimizer()
        
        # Sequence of calls:
        # 1. Optimizer LLM (generates improved prompt)
        # 2. Evaluate Baseline (Task 1)
        # 3. Judge Baseline (Task 1)
        # 4. Evaluate Baseline (Task 2)
        # 5. Judge Baseline (Task 2)
        # 6. Evaluate Candidate (Task 1)
        # 7. Judge Candidate (Task 1)
        # 8. Evaluate Candidate (Task 2)
        # 9. Judge Candidate (Task 2)
        
        # Setup:
        # Optimization result
        optimization_response = {"result": "Super Improved Prompt"}
        
        # Baseline scores (Average 5)
        baseline_task_1 = {"result": "Old Response 1"}
        baseline_judge_1 = {"result": "5"}
        baseline_task_2 = {"result": "Old Response 2"}
        baseline_judge_2 = {"result": "5"}
        
        # Candidate scores (Average 9)
        candidate_task_1 = {"result": "New Response 1"}
        candidate_judge_1 = {"result": "9"}
        candidate_task_2 = {"result": "New Response 2"}
        candidate_judge_2 = {"result": "9"}
        
        mock_run_llm.side_effect = [
            optimization_response,
            baseline_task_1, baseline_judge_1, baseline_task_2, baseline_judge_2,
            candidate_task_1, candidate_judge_1, candidate_task_2, candidate_judge_2
        ]
        
        result = await optimizer.optimize_prompt("test_key")
        
        self.assertEqual(result, "Super Improved Prompt")
        print("\nTest Promotion: Successful")

    @patch("core.polly.run_llm", new_callable=AsyncMock)
    async def test_optimize_prompt_rejection(self, mock_run_llm):
        """Test that a worse prompt is rejected."""
        optimizer = PollyOptimizer()
        
        # Optimization result
        optimization_response = {"result": "Bad Improved Prompt"}
        
        # Baseline scores (Average 8)
        # We need 4 calls for baseline (Task, Judge, Task, Judge)
        baseline_side_effects = [
            {"result": "res"}, {"result": "8"},
            {"result": "res"}, {"result": "8"}
        ]
        
        # Candidate scores (Average 4)
        candidate_side_effects = [
            {"result": "res"}, {"result": "4"},
            {"result": "res"}, {"result": "4"}
        ]
        
        mock_run_llm.side_effect = [optimization_response] + baseline_side_effects + candidate_side_effects
        
        result = await optimizer.optimize_prompt("test_key")
        
        self.assertIsNone(result)
        print("\nTest Rejection: Successful")

if __name__ == "__main__":
    unittest.main()
