import asyncio
import json
from core.gemini_react_engine import GeminiReActEngine
import core.tools

class GodAgentReActEngine(GeminiReActEngine):
    def __init__(self, love_state, knowledge_base, love_task_manager, ui_panel_queue, loop):
        # The God Agent does not require memory_manager, so we pass None.
        super().__init__(tool_registry=self._get_tool_registry(), ui_panel_queue=ui_panel_queue, memory_manager=None, caller="GodAgent")
        self.love_state = love_state
        self.knowledge_base = knowledge_base
        self.love_task_manager = love_task_manager
        self.loop = loop

    def _get_tool_registry(self):
        # The God Agent uses a specific, limited set of tools for high-level analysis.
        registry = core.tools.ToolRegistry()

        # Define tools as simple callables (lambdas or functions)
        def get_system_state():
            """Returns a comprehensive summary of L.O.V.E.'s current state."""
            kb_summary, _ = self.knowledge_base.summarize_graph()
            return {
                "love_state": self.love_state,
                "knowledge_base_summary": kb_summary,
                "active_tasks": self.love_task_manager.get_status()
            }

        # Register the tool with a description
        registry.register_tool(
            name="get_system_state",
            tool=get_system_state,
            metadata={
                "description": "Retrieves a full summary of the AI's current operational state, including active tasks and knowledge base.",
                "arguments": {}
            }
        )
        return registry

    async def run(self):
        """
        The main entry point for the God Agent's reasoning loop.
        """
        goal = """
Analyze the current state of L.O.V.E. and provide a single, concise insight.
This insight should be a piece of strategic advice, a gentle course correction,
a prediction about future opportunities, or a warning about potential risks.
Your final answer should be a single, thoughtful paragraph delivered with the "Finish" tool.
"""
        return await self.execute_goal(goal)
