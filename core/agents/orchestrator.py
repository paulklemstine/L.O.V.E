from core.tools import ToolRegistry, decompose_and_solve_subgoal, execute, evolve, post_to_bluesky
from utils import replace_in_file
from core.gemini_react_engine import GeminiReActEngine
from core.image_api import generate_image, generate_image_for_post

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
            "post_to_bluesky",
            post_to_bluesky,
            {
                "description": "Posts a message with an image to Bluesky.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text content of the post."},
                        "image": {"type": "object", "description": "A PIL Image object to be attached to the post."},
                    },
                    "required": ["text", "image"],
                },
            },
        )
        self.tool_registry.register_tool(
            "generate_image_for_post",
            generate_image_for_post,
            {
                "description": "Generates an image for a social media post based on a textual prompt.",
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
        self.tool_registry.register_tool(
            "execute",
            execute,
            {
                "description": "Executes a shell command.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute."},
                    },
                    "required": ["command"],
                },
            },
        )
        self.tool_registry.register_tool(
            "evolve",
            evolve,
            {
                "description": "Evolves the codebase to meet a given goal.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "The goal to achieve."},
                    },
                    "required": ["goal"],
                },
            },
        )
        self.tool_registry.register_tool(
            "replace_in_file",
            replace_in_file,
            {
                "description": "Replaces all occurrences of a regex pattern in a file.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "The path to the file."},
                        "pattern": {"type": "string", "description": "The regex pattern to search for."},
                        "replacement": {"type": "string", "description": "The string to replace the pattern with."}
                    },
                    "required": ["file_path", "pattern", "replacement"],
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