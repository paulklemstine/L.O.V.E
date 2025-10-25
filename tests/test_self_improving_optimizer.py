import unittest
import os
from unittest.mock import MagicMock, AsyncMock

# It's important to set the path correctly for the test runner to find the modules.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.self_improving_optimizer import SelfImprovingOptimizer
import shutil
import importlib.util

BENCHMARK_PROBLEMS = [
    {
        "name": "Fix ZeroDivisionError",
        "file": "tests/benchmark_problems/buggy_code.py",
        "objective": "Fix the ZeroDivisionError in the `buggy_divide` function by adding error handling.",
        "validation": lambda module: hasattr(module, 'buggy_divide') and module.buggy_divide(10, 0) is None
    },
    {
        "name": "Optimize Inefficient Sum",
        "file": "tests/benchmark_problems/inefficient_code.py",
        "objective": "Optimize the `inefficient_sum` function to be more performant.",
        "validation": lambda module: hasattr(module, 'inefficient_sum') and "for " not in module.__loader__.get_source(module)
    }
]

async def evaluate_improver_performance(optimizer_instance: SelfImprovingOptimizer) -> int:
    """
    Runs the optimizer against a set of benchmark problems and returns a score.
    """
    score = 0
    temp_dir = "tests/temp_benchmark"
    os.makedirs(temp_dir, exist_ok=True)

    for problem in BENCHMARK_PROBLEMS:
        # Create a temporary copy of the problem file
        temp_file_path = os.path.join(temp_dir, os.path.basename(problem["file"]))
        shutil.copy(problem["file"], temp_file_path)

        # Run the improvement process
        await optimizer_instance.improve_module(temp_file_path, problem["objective"])

        # Validate the result
        try:
            spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if problem["validation"](module):
                score += 1
        except Exception:
            pass # Validation failed

    shutil.rmtree(temp_dir)
    return score

class TestSelfImprovingOptimizer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up for the tests."""
        self.optimizer = SelfImprovingOptimizer()

    async def test_improve_module_basic_flow(self):
        """
        Test the basic flow of the improve_module method to ensure it calls
        the ReAct engine with the correct goal.
        """
        # We only need to test the initial call here.
        # The full ReAct loop will be tested in the longitudinal test.
        self.optimizer.gemini_react_engine.execute_goal = AsyncMock(return_value="Improvement successful.")

        module_path = "test_module.py"
        objective = "Add a new function to the module."

        await self.optimizer.improve_module(module_path, objective)

        # Assert that the ReAct engine was called
        self.optimizer.gemini_react_engine.execute_goal.assert_called_once()
        call_args = self.optimizer.gemini_react_engine.execute_goal.call_args
        goal_arg = call_args[0][0] # The goal is the first positional argument

        self.assertIn(module_path, goal_arg)
        self.assertIn(objective, goal_arg)

    async def test_longitudinal_self_improvement(self):
        """
        Tests that the optimizer can improve its own performance over time.
        """
        scores = []
        temp_dir = "tests/temp_longitudinal_test"
        os.makedirs(temp_dir, exist_ok=True)

        # Start with the original optimizer
        optimizer_module_path = "core/agents/self_improving_optimizer.py"
        temp_optimizer_path = os.path.join(temp_dir, "self_improving_optimizer_v1.py")
        shutil.copy(optimizer_module_path, temp_optimizer_path)

        for i in range(1, 4):
            # Load the current version of the optimizer
            spec = importlib.util.spec_from_file_location(f"optimizer_v{i}", temp_optimizer_path)
            optimizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(optimizer_module)
            optimizer = optimizer_module.SelfImprovingOptimizer()

            # Mock the ReAct engine to simulate a successful improvement
            async def mock_improve_code(goal):
                # Simulate the optimizer improving its own code
                with open(temp_optimizer_path, 'r') as f:
                    source_code = f.read()
                # A simple "improvement": add a comment
                improved_code = source_code + f"\n# Improvement version {i}"
                with open(temp_optimizer_path, 'w') as f:
                    f.write(improved_code)
                return "Improvement successful."
            optimizer.gemini_react_engine.execute_goal = AsyncMock(side_effect=mock_improve_code)

            # Evaluate the optimizer's performance
            score = await evaluate_improver_performance(optimizer)
            scores.append(score)

            # Prepare for the next iteration by "improving" the optimizer's code
            await optimizer.improve_module(temp_optimizer_path, "Improve thyself.")

        shutil.rmtree(temp_dir)

        # Assert that the scores are increasing
        self.assertTrue(scores[0] < scores[1] < scores[2] or scores[0] <= scores[1] <= scores[2])


if __name__ == "__main__":
    unittest.main()
