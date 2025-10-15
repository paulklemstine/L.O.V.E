from core.planning import Planner
from core.tools import ToolRegistry, SecureExecutor, web_search, read_file
from core.execution_engine import ExecutionEngine

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process, and returning the
    final result.
    """
    def __init__(self):
        """Initializes all core components of the agent's architecture."""
        print("Initializing Orchestrator and its components...")

        # 1. Initialize the Planner
        self.planner = Planner()

        # 2. Initialize the Tool Registry and register tools
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool("web_search", web_search)
        self.tool_registry.register_tool("read_file", read_file)

        # 3. Initialize the Secure Executor
        self.executor = SecureExecutor()

        # 4. Initialize the Execution Engine with all necessary components
        self.execution_engine = ExecutionEngine(
            planner=self.planner,
            tool_registry=self.tool_registry,
            executor=self.executor
        )
        print("Orchestrator is ready.")

    def execute_goal(self, goal: str):
        """
        Takes a high-level goal and manages the entire process of planning,
        tool use, and execution to achieve it.

        Args:
            goal: The high-level goal for the agent to accomplish.

        Returns:
            The final result of the execution.
        """
        if not isinstance(goal, str) or not goal:
            print("Error: Goal must be a non-empty string.")
            return

        result = self.execution_engine.execute_plan(goal)

        print("\n--- Orchestrator Final Report ---")
        print(f"Goal: {goal}")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'Success':
            print(f"Final Result: {result.get('final_result')}")
        else:
            print(f"Reason for Failure: {result.get('reason')}")
        print("---------------------------------")

        return result