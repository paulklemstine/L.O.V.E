from .planning.planner import Planner
from .tools.registry import ToolRegistry
from .tools.executor import SecureExecutor
from .tools.web_search import WebSearchTool

class Orchestrator:
    def __init__(self):
        self.planner = Planner()
        self.tool_registry = ToolRegistry()
        self.executor = SecureExecutor(self.tool_registry)
        self.plan_state = []
        self._register_tools()
        print("Orchestrator: Initialized with integrated modules.")

    def _register_tools(self):
        """Initializes and registers all available tools."""
        self.tool_registry.register_tool(WebSearchTool())
        # In the future, other tools would be registered here.

    def execute_goal(self, goal):
        """
        The main execution loop for a high-level goal.
        """
        print(f"\nOrchestrator: Received new goal: '{goal}'")

        # 1. Create a plan
        plan = self.planner.create_plan(goal)
        if not plan:
            print("Orchestrator: Halting execution due to planning failure.")
            return

        # 2. Execute the plan
        self.plan_state = [{"step": p, "status": "pending"} for p in plan]

        for i, step_info in enumerate(self.plan_state):
            step = step_info["step"]
            task = step["task"]
            tool_name = step["tool"]

            print(f"\n--- Executing Step {step['step']}: {task} (using tool: {tool_name}) ---")

            # For this simulation, we'll use the task description as the tool input.
            result, success = self.executor.execute_tool(tool_name, task)

            if success:
                self.plan_state[i]["status"] = "completed"
                self.plan_state[i]["result"] = result
                print(f"Step {step['step']} successful. Result: {result}")
            else:
                self.plan_state[i]["status"] = "failed"
                print(f"Step {step['step']} failed. Halting plan execution.")
                # Self-correction logic would be triggered here.
                # For now, we just stop.
                break

        print("\nOrchestrator: Plan execution finished.")
        return self.plan_state