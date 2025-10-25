from core.planning import Planner
from core.execution_engine import ExecutionEngine
from core.tools import ToolRegistry, SecureExecutor, decompose_and_solve_subgoal
from core.gemini_react_engine import GeminiReActEngine
from core.image_api import generate_image
from core.knowledge_graph.graph import KnowledgeGraph

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process, and returning the
    final result. It uses the GeminiReActEngine by default and falls back
    to a legacy planner if the Gemini CLI is unavailable.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        """Initializes all core components of the agent's architecture."""
        print("Initializing Orchestrator and its components...")

        # 1. Initialize the Knowledge Graph
        self.kg = knowledge_graph if knowledge_graph else KnowledgeGraph()

        # 2. Initialize the Tool Registry and register tools
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

        # 3. Initialize the Planners and Execution Engine
        self.react_engine = None
        self.legacy_planner = Planner(knowledge_graph=self.kg)
        self.secure_executor = SecureExecutor(knowledge_graph=self.kg, llm_api=None) # llm_api can be None if not used for knowledge extraction in legacy mode
        self.evil_state = {"knowledge_base": {"network_map": {}}} # A default state
        self.execution_engine = ExecutionEngine(
            planner=self.legacy_planner,
            tool_registry=self.tool_registry,
            executor=self.secure_executor,
            evil_state=self.evil_state
        )

        try:
            # The GeminiReActEngine will raise FileNotFoundError if gemini-cli is not found.
            self.react_engine = GeminiReActEngine(self.tool_registry)
            print("GeminiReActEngine initialized successfully.")
        except FileNotFoundError:
            print("Warning: gemini-cli not found. Falling back to legacy planner.")
            # We can optionally register more tools for the legacy planner here if needed.

        print("Orchestrator is ready.")

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal and manages the entire process
        of planning, tool use, and execution to achieve it.
        """
        if not isinstance(goal, str) or not goal:
            print("Error: Goal must be a non-empty string.")
            return

        result = None
        if self.react_engine:
            print("Executing goal with GeminiReActEngine...")
            try:
                result = await self.react_engine.execute_goal(goal)
            except Exception as e:
                print(f"Error executing with GeminiReActEngine: {e}. Falling back to legacy planner.")
                # Ensure we fall back if the engine fails for reasons other than init.
                result = None

        if result is None:
            print("Executing goal with legacy planner...")
            plan = self.legacy_planner.create_plan(goal)
            if not plan:
                result = "Failed to create a plan."
            else:
                final_step_result = await self.execution_engine.execute_plan(plan, self.tool_registry)
                result = final_step_result

        print("\n--- Orchestrator Final Report ---")
        print(f"Goal: {goal}")
        print(f"Status: Success")
        print(f"Final Result: {result}")
        print("---------------------------------")

        return result