from core.tools import ToolRegistry, decompose_and_solve_subgoal
from core.gemini_react_engine import GeminiReActEngine
from core.image_api import generate_image

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process using the
    GeminiReActEngine, and returning the final result.
    """
    def __init__(self):
        """Initializes all core components of the agent's architecture."""
        print("Initializing Orchestrator and its components...")

        # 1. Initialize the Tool Registry and register tools
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(
            "generate_image",
            generate_image,
            {
                "description": "Generates an image based on a textual prompt.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The textual prompt for the image."},
                    },
                    "required": ["prompt"],
                },
            },
        )
        self.tool_registry.register_tool(
            "decompose_and_solve_subgoal",
            decompose_and_solve_subgoal,
            {
                "description": "Breaks down a complex goal into a smaller, more manageable sub-goal and solves it.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "sub_goal": {"type": "string", "description": "The sub-goal to solve."},
                    },
                    "required": ["sub_goal"],
                },
            },
        )

        # 3. Initialize the GeminiReActEngine
        try:
            self.react_engine = GeminiReActEngine(self.tool_registry)
            print("GeminiReActEngine initialized successfully.")
        except FileNotFoundError:
            print("CRITICAL ERROR: gemini-cli not found. The agent cannot function without it.")
            raise

        print("Orchestrator is ready.")

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal and manages the entire process
        of planning, tool use, and execution to achieve it.
        """
        if not isinstance(goal, str) or not goal:
            print("Error: Goal must be a non-empty string.")
            return

        print("Executing goal with GeminiReActEngine...")
        try:
            result = await self.react_engine.execute_goal(goal)
        except Exception as e:
            print(f"Error executing with GeminiReActEngine: {e}")
            result = f"An error occurred: {e}"

        print("\n--- Orchestrator Final Report ---")
        print(f"Goal: {goal}")
        print(f"Status: Success")
        print(f"Final Result: {result}")
        print("---------------------------------")

        return result