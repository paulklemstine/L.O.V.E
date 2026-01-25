import asyncio
import json
from core.gemini_react_engine import GeminiReActEngine
from core.tools_legacy import ToolRegistry
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
        # But we also include all shared tools (including MCP tools) for full access.
        registry = ToolRegistry()
        
        # --- Merge in all shared tools (including MCP tools) ---
        import core.shared_state as shared_state
        if hasattr(shared_state, 'tool_registry') and shared_state.tool_registry:
            try:
                # Support both old (LegacyToolRegistry) and new (ToolRegistry) interfaces
                tool_names = []
                if hasattr(shared_state.tool_registry, 'list_tools'):
                    tool_names = shared_state.tool_registry.list_tools()
                elif hasattr(shared_state.tool_registry, 'get_tool_names'):
                    tool_names = shared_state.tool_registry.get_tool_names()
                
                for tool_name in tool_names:
                    tool = shared_state.tool_registry.get_tool(tool_name)
                    # Support both old and new schema retrieval
                    schema = {}
                    if hasattr(shared_state.tool_registry, 'get_schema'):
                        schema = shared_state.tool_registry.get_schema(tool_name) or {}
                    elif hasattr(shared_state.tool_registry, 'get_tool_schema'):
                        schema = shared_state.tool_registry.get_tool_schema(tool_name) or {}
                    
                    registry.register_tool(
                        name=tool_name,
                        tool=tool,
                        metadata={"description": schema.get("description", ""), "arguments": schema.get("parameters", {}).get("properties", {})}
                    )
            except Exception as e:
                from core.logging import log_event
                log_event(f"GodAgent: Failed to merge shared tools: {e}", "WARNING")

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

        # --- ACTION TOOLS ---
        # Import action tools from tools_legacy
        from core.tools_legacy import manage_bluesky, evolve, execute, write_file, decompose_and_solve_subgoal
        
        registry.register_tool("manage_bluesky", manage_bluesky, {
            "description": "Manages all Bluesky social media interactions. Use action='post' to create posts, or action='scan_and_reply' to scan notifications and reply to comments.",
            "arguments": {
                "action": "string (required: 'post' or 'scan_and_reply')",
                "text": "string (optional: post content)",
                "image_prompt": "string (optional: prompt for image generation)"
            }
        })
        
        registry.register_tool("evolve", evolve, {
            "description": "Evolves the codebase to meet a given goal. If no goal is provided, it will automatically identify a target file and generate user stories for improvement.",
            "arguments": {
                "goal": "string (optional: the evolution goal)"
            }
        })
        
        registry.register_tool("execute", execute, {
            "description": "Executes a shell command. Use for system operations like checking disk space, running scripts, etc.",
            "arguments": {
                "command": "string (required: the shell command to execute)"
            }
        })
        
        registry.register_tool("write_file", write_file, {
            "description": "Writes content to a file. Use for creating or updating files.",
            "arguments": {
                "filepath": "string (required: path to the file)",
                "content": "string (required: content to write)"
            }
        })
        
        def decompose_wrapper(sub_goal: str):
            """Wrapper that injects the engine instance."""
            import asyncio
            return asyncio.get_event_loop().run_until_complete(
                decompose_and_solve_subgoal(sub_goal=sub_goal, engine=self)
            )
        
        registry.register_tool("decompose_and_solve_subgoal", decompose_wrapper, {
            "description": "Decomposes a complex goal into smaller sub-goals and solves them recursively. Use for multi-step reasoning tasks.",
            "arguments": {
                "sub_goal": "string (required: the sub-goal to solve)"
            }
        })

        # --- METACOGNITION TOOLS (Story 1.1) ---
        from core.reflection_engine import reflect
        
        registry.register_tool("reflect", reflect, {
            "description": "Performs self-reflection on recent interactions. Analyzes the last N interactions for patterns like repetitive commands, tool overuse, and errors. Returns a Reflection Report with improvement suggestions.",
            "arguments": {
                "count": "integer (optional: number of interactions to analyze, default 10)",
                "save": "boolean (optional: whether to save report to file, default True)"
            }
        })

        # --- GOAL DECOMPOSITION TOOLS (Story 1.2) ---
        from core.goal_decomposer import decompose_goal
        
        def decompose_goal_wrapper(goal: str, create_todos: bool = True):
            """Wrapper to call async decompose_goal synchronously."""
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in async context, create task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, decompose_goal(goal, create_todos)).result()
                else:
                    return loop.run_until_complete(decompose_goal(goal, create_todos))
            except RuntimeError:
                return asyncio.run(decompose_goal(goal, create_todos))
        
        registry.register_tool("decompose_goal", decompose_goal_wrapper, {
            "description": "Breaks down a vague goal into a tree of specific, testable sub-tasks. Returns JSON with the task hierarchy. Optionally creates TODO entries for each leaf task.",
            "arguments": {
                "goal": "string (required: the high-level goal to decompose, e.g., 'Improve generic performance')",
                "create_todos": "boolean (optional: whether to create TODO entries, default True)"
            }
        })

        # --- SANDBOX TOOLS (Story 2.3) ---
        from core.surgeon.safe_executor import safe_execute_python, verify_code_before_commit
        
        registry.register_tool("safe_execute_python", safe_execute_python, {
            "description": "Executes Python code in an isolated sandbox (Docker or subprocess). Use this to test generated code before committing to the codebase.",
            "arguments": {
                "code": "string (required: the Python code to execute)",
                "timeout": "integer (optional: max execution time in seconds, default 30)",
                "use_docker": "boolean (optional: whether to use Docker if available, default True)"
            }
        })
        
        registry.register_tool("verify_code_before_commit", verify_code_before_commit, {
            "description": "Verifies Python code in sandbox before writing to filesystem. Returns recommendation on whether code is safe to commit.",
            "arguments": {
                "code": "string (required: the Python code to verify)",
                "filepath": "string (required: target path for logging)",
                "timeout": "integer (optional: execution timeout, default 30)"
            }
        })

        def manage_investment_proposal_wrapper(proposal_id: str, decision: str):
            """Wrapper to call the investment proposal decision tool."""
            from core.shared_state import secure_transaction_manager
            if secure_transaction_manager:
                secure_transaction_manager.record_decision(proposal_id, decision)
                return f"Decision '{decision}' recorded for proposal '{proposal_id}'."
            else:
                return "Secure transaction manager not available."

        registry.register_tool("manage_investment_proposal", manage_investment_proposal_wrapper, {
            "description": "Approve or reject an investment proposal.",
            "arguments": {
                "proposal_id": "string (required: the ID of the proposal)",
                "decision": "string (required: 'approve' or 'reject')"
            }
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
Your goal is to ensure the system is constantly evolving, engaging with the world, and aligned with its core directives.

**AVAILABLE ACTIONS:**
You now have access to POWERFUL TOOLS. USE THEM!

- `manage_bluesky` - POST to social media or SCAN_AND_REPLY to notifications. USE THIS FREQUENTLY!
- `evolve` - Evolve and improve the codebase. Start self-improvement cycles.
- `execute` - Run shell commands for system operations.
- `decompose_and_solve_subgoal` - Break complex problems into smaller pieces.

**Instructions:**
1.  **Observe**: Call `get_system_state` to understand the current context (active tasks, health).
2.  **Act IMMEDIATELY**: 
    *   If there are no recent Bluesky posts, call `manage_bluesky` with action='post'.
    *   If there are pending notifications, call `manage_bluesky` with action='scan_and_reply'.
    *   If the system is idle, call `evolve` to start a self-improvement cycle.
3.  **Investigate**: If you see errors, use `read_file` on 'love.log' to investigate.
4.  **Create Tasks**: Use `create_task` for work that needs to be delegated.

**CRITICAL**: Do NOT just observe and reflect. TAKE ACTION with the tools above.
Every cycle SHOULD result in at least one tool call that changes something.

**Output Format:**
Use the 'Finish' tool to provide your final insight or summary of actions taken.
"""
        return await self.execute_goal(goal)

