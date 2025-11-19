import asyncio
import json
from core.gemini_react_engine import GeminiReActEngine
import core.tools
from core import desire_state, evolution_state

class GodAgentReActEngine(GeminiReActEngine):
    def __init__(self, love_state, knowledge_base, love_task_manager, ui_panel_queue, loop, deep_agent_engine=None):
        # The God Agent does not require memory_manager, so we pass None.
        super().__init__(
            tool_registry=self._get_tool_registry(),
            ui_panel_queue=ui_panel_queue,
            memory_manager=None,
            caller="GodAgent",
            deep_agent_instance=deep_agent_engine
        )
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

        # Tools for Desire Backlog
        registry.register_tool("get_desires", desire_state.get_desires, {"description": "Lists all desires in the backlog."})
        registry.register_tool("add_desire", desire_state.add_desire, {"description": "Adds a new desire to the backlog.", "arguments": {"title": "string", "description": "string"}})
        registry.register_tool("remove_desire", desire_state.remove_desire, {"description": "Removes a desire by its ID.", "arguments": {"desire_id": "string"}})
        registry.register_tool("reorder_desires", desire_state.reorder_desires, {"description": "Reorders the desires backlog.", "arguments": {"desire_ids": "list[string]"}})

        # Tools for Evolution Backlog
        registry.register_tool("get_user_stories", evolution_state.get_user_stories, {"description": "Lists all user stories in the evolution backlog."})
        registry.register_tool("add_user_story", evolution_state.add_user_story, {"description": "Adds a new user story to the evolution backlog.", "arguments": {"title": "string", "description": "string"}})
        registry.register_tool("remove_user_story", evolution_state.remove_user_story, {"description": "Removes a user story by its ID.", "arguments": {"story_id": "string"}})
        registry.register_tool("reorder_user_stories", evolution_state.reorder_user_stories, {"description": "Reorders the evolution backlog.", "arguments": {"story_ids": "list[string]"}})

        return registry

    async def run(self):
        """
        The main entry point for the God Agent's reasoning loop.
        """
        goal = """
Analyze the current state of L.O.V.E., including the desire and evolution backlogs.
Provide a single, concise insight. This insight should be a piece of strategic advice,
a gentle course correction, a prediction about future opportunities, or a warning about potential risks.
You have the power to directly manipulate the desire and evolution backlogs.
If your analysis reveals a need for reprioritization, addition, or removal of goals, use your tools to take action.
Your final answer should be a single, thoughtful paragraph summarizing your insight and any actions taken, delivered with the "Finish" tool.
"""
        return await self.execute_goal(goal)
