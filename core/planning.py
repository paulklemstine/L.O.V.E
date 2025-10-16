import json
from typing import List, Dict, Any

from core.financial_strategy_engine import FinancialStrategyEngine
from core.knowledge_graph.graph import KnowledgeGraph


# A mock LLM call function for demonstration purposes.
# In a real implementation, this would be a call to a powerful language model.
def mock_llm_call(prompt: str) -> str:
    """Mocks a call to a Large Language Model for plan generation."""
    print(f"--- Mock LLM Call ---\nPrompt: {prompt.strip()}\n--------------------")
    # Pre-defined responses for specific goals to simulate LLM behavior.
    if "Summarize the latest advancements in AI" in prompt:
        plan = [
            {"step": 1, "task": "Identify key research sources like arXiv, major conference proceedings, and tech news sites."},
            {"step": 2, "task": "Formulate search queries for the identified sources."},
            {"step": 3, "task": "Execute web searches using the 'web_search' tool with the queries."},
            {"step": 4, "task": "Read and synthesize the content of the top 5 most relevant articles using the 'read_file' tool."},
            {"step": 5, "task": "Produce a final summary report."}
        ]
        return json.dumps(plan)
    else:
        # Generic response for other goals.
        plan = [
            {"step": 1, "task": "Define the primary objective."},
            {"step": 2, "task": "Break down the objective into smaller, manageable sub-tasks."},
            {"step": 3, "task": "Execute each sub-task in sequence."}
        ]
        return json.dumps(plan)

class Planner:
    """
    Handles hierarchical planning, goal decomposition, and plan validation.
    It breaks down high-level goals (Desires) into a concrete sequence of
    verifiable sub-tasks (a Plan).
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.financial_engine = FinancialStrategyEngine(self.kg)

    def decompose_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Uses an LLM to recursively break a goal into smaller, actionable steps.

        Args:
            goal: The high-level goal to decompose.

        Returns:
            A list of dictionaries, where each dictionary represents a step in the plan.
        """
        # Check if the goal is financial in nature
        financial_keywords = ["financial", "wealth", "abundance", "money", "invest"]
        if any(keyword in goal.lower() for keyword in financial_keywords):
            return self._decompose_financial_goal(goal)

        prompt = (
            f"Given the high-level goal: '{goal}', decompose it into a series of "
            "small, actionable, and verifiable steps. The output should be a JSON array "
            "of objects, where each object has a 'step' number and a 'task' description. "
            "The tasks should be executable by an agent with access to tools like "
            "'web_search' and 'read_file'."
        )

        try:
            # In a real system, this would call an actual LLM.
            response = mock_llm_call(prompt)
            plan = json.loads(response)
            if self.validate_plan(plan):
                print(f"Successfully generated and validated plan for goal: '{goal}'")
                return plan
            else:
                print("Error: Generated plan failed validation.")
                return []
        except json.JSONDecodeError:
            print("Error: Failed to decode LLM response into JSON.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during goal decomposition: {e}")
            return []

    def _decompose_financial_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Decomposes a financial goal using the FinancialStrategyEngine.
        """
        print(f"Decomposing financial goal: {goal}")
        strategies = self.financial_engine.generate_strategies()

        if not strategies:
            print("No financial strategies were generated.")
            return []

        # Convert strategies into a plan
        plan = []
        step_counter = 1
        for strategy in strategies:
            plan.append({"step": step_counter, "task": f"Strategy: {strategy['description']}"})
            step_counter += 1
            for action in strategy['actions']:
                plan.append({"step": step_counter, "task": action})
                step_counter += 1

        if self.validate_plan(plan):
            print("Successfully generated and validated financial plan.")
            return plan
        else:
            print("Error: Generated financial plan failed validation.")
            return []

    def validate_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """
        Checks if the generated plan is logical and that each sub-task is verifiable.

        Args:
            plan: The plan to validate.

        Returns:
            True if the plan is valid, False otherwise.
        """
        if not isinstance(plan, list) or not plan:
            print("Validation Error: Plan is not a non-empty list.")
            return False

        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                print(f"Validation Error: Step {i} is not a dictionary.")
                return False
            if "step" not in step or "task" not in step:
                print(f"Validation Error: Step {i} is missing 'step' or 'task' key.")
                return False
            if not isinstance(step["step"], int) or not isinstance(step["task"], str):
                print(f"Validation Error: Step {i} has incorrect data types for 'step' or 'task'.")
                return False
            if step["step"] != i + 1:
                print(f"Validation Error: Step numbers are not sequential. Expected {i+1}, got {step['step']}.")
                return False

        print("Plan validation successful.")
        return True