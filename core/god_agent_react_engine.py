import asyncio
import json
from core.gemini_react_engine import GeminiReActEngine
import core.tools
from core import desire_state, evolution_state

class GodAgentReActEngine(GeminiReActEngine):
    def __init__(self, love_state, knowledge_base, love_task_manager, ui_panel_queue, loop, deep_agent_engine=None, memory_manager=None):
        super().__init__(
            tool_registry=self._get_tool_registry(),
            ui_panel_queue=ui_panel_queue,
            memory_manager=memory_manager,
            caller="GodAgent",
            deep_agent_instance=deep_agent_engine
        )
        self.love_state = love_state
        self.knowledge_base = knowledge_base
        self.memory_manager = memory_manager
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

        # --- KB Tools ---
        from core.kb_tools import query_knowledge_base, search_memories, get_kb_summary
        
        def query_kb_wrapper(node_type: str, limit: int = 10):
            return query_knowledge_base(node_type, limit, self.knowledge_base)
            
        def search_memories_wrapper(query: str, top_k: int = 3):
            return search_memories(query, top_k, self.memory_manager)
            
        def get_kb_summary_wrapper(max_tokens: int = 512):
            return get_kb_summary(self.knowledge_base, max_tokens)

        registry.register_tool("query_knowledge_base", query_kb_wrapper, {
            "description": "Queries the main knowledge graph by node type (e.g., 'task', 'talent', 'host').",
            "arguments": {"node_type": "string", "limit": "integer"}
        })
        
        registry.register_tool("search_memories", search_memories_wrapper, {
            "description": "Performs semantic search on the memory system using FAISS.",
            "arguments": {"query": "string", "top_k": "integer"}
        })
        
        registry.register_tool("get_kb_summary", get_kb_summary_wrapper, {
            "description": "Returns a high-level summary of the knowledge base.",
            "arguments": {"max_tokens": "integer"}
        })

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

        # --- Action Tools ---
        def create_task_wrapper(description: str):
            """Creates a new task in the LoveTaskManager."""
            task_id = self.love_task_manager.add_task(description)
            return f"Task created with ID: {task_id}"

        def read_file_wrapper(file_path: str):
            """Reads the content of a file."""
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        registry.register_tool("create_task", create_task_wrapper, {
            "description": "Creates a new task for the system to execute.",
            "arguments": {"description": "string"}
        })

        registry.register_tool("read_file", read_file_wrapper, {
            "description": "Reads a file from the filesystem. Use this to check logs or code.",
            "arguments": {"file_path": "string"}
        })

        return registry

    async def run(self):
        """
        The main entry point for the God Agent's reasoning loop.
        """
        # Get recent memories/insights to inject into context
        recent_insights = []
        if self.memory_manager:
            # This is a simplification; ideally we'd query the memory manager
            pass

        goal = """
You are the God Agent, the high-level orchestrator of the L.O.V.E. system.
Your goal is to ensure the system is constantly evolving, stable, and aligned with its core directives.

**Instructions:**
1.  **Observe**: Call `get_system_state` to understand the current context (active tasks, health).
2.  **Reflect**: Check `get_desires` or `get_user_stories` if the system is idle.
3.  **Investigate**: If you see errors in the logs or stuck tasks, use `read_file` (e.g., 'love.log') to investigate.
4.  **Plan & Act**:
    *   If a critical issue is found, create a task to fix it using `create_task`.
    *   If the system is idle, pick a desire/story and create a task to implement it.
    *   If everything is running smoothly, provide a strategic insight.

**Output Format:**
Use the 'Finish' tool to provide your final insight or summary of actions taken.
"""
        return await self.execute_goal(goal)

