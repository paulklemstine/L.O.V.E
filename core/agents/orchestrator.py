from core.gemini_cli_wrapper import GeminiCLIWrapper
from core.gemini_react_engine import GeminiReActEngine
from core.tools import ToolRegistry, decompose_and_solve_subgoal
from core.image_api import generate_image
from core.knowledge_graph.graph import KnowledgeGraph

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process, and returning the
    final result.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        """Initializes all core components of the agent's architecture."""
        print("Initializing Orchestrator and its components...")

        # 1. Initialize the Knowledge Graph
        self.kg = knowledge_graph if knowledge_graph else KnowledgeGraph()

        # 2. Initialize the Tool Registry and register tools
        self.tool_registry = ToolRegistry()

        # 3. Initialize the Gemini ReAct Engine
        self.gemini_cli_wrapper = GeminiCLIWrapper()
        self.react_engine = GeminiReActEngine(self.gemini_cli_wrapper, self.tool_registry)
        # self.tool_registry.register_tool("web_search", web_search)
        # self.tool_registry.register_tool("read_file", read_file)
        self.tool_registry.register_tool("generate_image", generate_image)
        self.tool_registry.register_tool("decompose_and_solve_subgoal", decompose_and_solve_subgoal)

        print("Orchestrator is ready.")

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal and manages the entire process
        of planning, tool use, and execution to achieve it.
        """
        if not isinstance(goal, str) or not goal:
            print("Error: Goal must be a non-empty string.")
            return

        result = await self.react_engine.execute_goal(goal)

        print("\n--- Orchestrator Final Report ---")
        print(f"Goal: {goal}")
        print(f"Status: Success")
        print(f"Final Result: {result}")
        print("---------------------------------")

        return result