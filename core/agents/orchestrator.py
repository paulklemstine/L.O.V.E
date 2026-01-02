import json
import asyncio
import re
from typing import Dict, List, Any

# Local, dynamic imports for specialist agents
from core.agents.analyst_agent import AnalystAgent
from core.agents.metacognition_agent import MetacognitionAgent
from core.agents.talent_agent import TalentAgent
from core.agents.web_automation_agent import WebAutomationAgent
from core.agents.memory_folding_agent import MemoryFoldingAgent
from core.agents.unified_reasoning_agent import UnifiedReasoningAgent
from core.llm_api import run_llm # Using a direct LLM call for planning
# Migrated from tools_legacy to new modules (Story 1.4)
from core.legacy_compat import ToolRegistry
from core.secure_executor import SecureExecutor
from core.tools import read_file, write_file  # Import commonly needed tools
from core.goal_decomposer import GoalDecomposer, GoalTree, DecomposedTask, TaskStatus, ExecutionMode
# MCP Integration (Epic 2: Story 2.1)
from core.mcp_adapter import get_all_mcp_langchain_tools, convert_mcp_to_langchain_tools
from core.logging import log_event
# Temporary: talent_scout still in tools_legacy until full migration
try:
    from core.tools_legacy import talent_scout
except ImportError:
    talent_scout = None
from core.image_api import generate_image

# Keep the old function for fallback compatibility as requested
async def solve_with_agent_team(task_description: str) -> str:
    import core.shared_state as shared_state
    from core.agent_framework_manager import create_and_run_workflow
    orchestrator = Orchestrator(shared_state.memory_manager)
    result = await create_and_run_workflow(task_description, orchestrator.tool_registry)
    return str(result)


class Orchestrator:
    """
    The Supervisor agent responsible for receiving high-level goals,
    decomposing them into a plan, and orchestrating a team of specialist
    agents to execute the plan.
    """
    def __init__(self, memory_manager):
        """Initializes the Supervisor and its registry of specialist agents."""
        print("Initializing Supervisor Orchestrator...")
        self.tool_registry = ToolRegistry()
        # Note: CodeGenerationAgent was removed as it doesn't exist
        self.specialist_registry = {
            "AnalystAgent": AnalystAgent,
            "TalentAgent": TalentAgent,
            "WebAutomationAgent": WebAutomationAgent,
            "MemoryFoldingAgent": MemoryFoldingAgent,
        }
        self.memory_manager = memory_manager
        self.metacognition_agent = MetacognitionAgent(self.memory_manager)
        self.goal_decomposer = GoalDecomposer(memory_manager=self.memory_manager)
        self.tool_registry = ToolRegistry()
        self.secure_executor = SecureExecutor()
        self.goal_counter = 0
        
        # Initialize MCP (Epic 2: Story 2.1)
        self.mcp_manager = self._initialize_mcp()
        
        self._register_tools()
        self._register_mcp_tools()  # Epic 2: Register MCP tools
        print("Supervisor Orchestrator is ready.")

    def _register_tools(self):
        """Registers all available tools and agents in the ToolRegistry."""
        # Register specialist agents as tools
        for name, agent_class in self.specialist_registry.items():
            # Create a wrapper function to make the agent compatible with the tool registry
            async def agent_wrapper(task_details: Dict) -> Dict:
                agent_instance = agent_class()
                return await agent_instance.execute_task(task_details)

            self.tool_registry.register_tool(
                name=name,
                tool=agent_wrapper,
                metadata={
                    "description": agent_class.__doc__ or f"The {name} specialist agent.",
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "task_details": {"type": "object", "description": "The specific parameters for the agent's task."}
                        },
                        "required": ["task_details"]
                    }
                }
            )

        # Register standalone tools
        self.tool_registry.register_tool(
            name="talent_scout",
            tool=talent_scout,
            metadata={
                "description": "Scouts for talented individuals based on a query.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query for talent."}
                    },
                    "required": ["query"]
                }
            }
        )
        self.tool_registry.register_tool(
            name="generate_image",
            tool=generate_image,
            metadata={
                "description": "Generates an image from a textual prompt.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The textual prompt to generate an image from."}
                    },
                    "required": ["prompt"]
                }
            }
        )
    
    def _initialize_mcp(self):
        """
        Initializes the MCPManager for MCP tool integration.
        Epic 2: Story 2.1 - Auto-discovery of MCP servers.
        
        Returns:
            MCPManager instance or None if initialization fails
        """
        try:
            from mcp_manager import MCPManager
            
            # Create a simple console interface for MCP logging
            class MCPConsole:
                def print(self, msg):
                    log_event(f"[MCP] {msg}", "DEBUG")
            
            mcp_manager = MCPManager(MCPConsole())
            
            # Log available servers
            server_configs = getattr(mcp_manager, 'server_configs', {})
            if server_configs:
                log_event(f"MCP servers discovered: {list(server_configs.keys())}", "INFO")
            else:
                log_event("No MCP servers configured in mcp_servers.json", "WARNING")
            
            return mcp_manager
            
        except ImportError as e:
            log_event(f"MCPManager not available: {e}", "WARNING")
            return None
        except Exception as e:
            log_event(f"Failed to initialize MCP: {e}", "ERROR")
            return None
    
    def _register_mcp_tools(self):
        """
        Registers all available MCP tools as LangChain tools in the registry.
        Epic 2: Story 2.1 - Tool wrapping and registry population.
        """
        if not self.mcp_manager:
            log_event("MCP not initialized, skipping MCP tool registration", "DEBUG")
            return
        
        try:
            # Get all MCP tools from all configured servers
            mcp_tools = get_all_mcp_langchain_tools(self.mcp_manager)
            
            if mcp_tools:
                # Register all MCP tools in our registry
                for tool in mcp_tools:
                    try:
                        self.tool_registry.register(tool)
                        log_event(f"Registered MCP tool: {tool.name}", "DEBUG")
                    except Exception as e:
                        log_event(f"Failed to register MCP tool {tool.name}: {e}", "WARNING")
                
                log_event(f"Registered {len(mcp_tools)} MCP tools in ToolRegistry", "INFO")
            else:
                log_event("No MCP tools available (servers may not be running)", "DEBUG")
                
        except Exception as e:
            log_event(f"Error registering MCP tools: {e}", "ERROR")

    async def _classify_goal(self, goal: str) -> str:
        """
        Uses an LLM to classify the goal into 'Procedural' or 'Open-Ended'.
        """
        try:
            classification = await run_llm(prompt_key="orchestrator_goal_classification", prompt_vars={"goal": goal}, force_model=None)
            return (classification.get("result") or "").strip()
        except Exception as e:
            print(f"Error during goal classification: {e}")
            return "Procedural" # Default to procedural on error

    async def _generate_plan(self, goal: str) -> List[Dict]:
        """
        Uses an LLM to decompose a high-level goal into a structured,
        step-by-step plan for specialist agents.
        """
        print(f"Supervisor: Generating plan for goal: {goal}")

        available_tools = self.tool_registry.get_formatted_tool_metadata()
        try:
            response = await run_llm(prompt_key="orchestrator_plan_generation", prompt_vars={"available_tools": available_tools, "goal": goal}, is_source_code=False, force_model=None)
            # Clean the response to extract only the JSON part
            json_match = re.search(r'\[.*\]', response.get("result", ""), re.DOTALL)
            if not json_match:
                print(f"Supervisor: Failed to extract JSON plan from LLM response: {response}")
                return []
            plan_str = json_match.group(0)
            plan = json.loads(plan_str)

            # Emit plan generation event
            event_payload = {'event_type': 'plan_generated', 'goal': goal, 'plan': plan}
            asyncio.create_task(self.metacognition_agent.execute_task(event_payload))

            return plan
        except Exception as e:
            print(f"Supervisor: Error during plan generation: {e}")
            return []

    async def execute_goal(self, goal: str) -> Any:
        """
        Asynchronously takes a high-level goal, decomposes it into a hierarchical
        task tree, and executes it using depth-first traversal.
        
        Now uses GoalDecomposer for recursive decomposition (Story 1.1).
        """
        print(f"\n--- Supervisor received new goal: {goal} ---")

        self.goal_counter += 1
        # Every 5 goals, trigger the memory folding agent.
        if self.goal_counter % 5 == 0:
            print("--- Supervisor: Goal threshold reached. Triggering autonomous memory folding. ---")
            folding_goal = "Fold long memory chains to distill insights."
            # Run this as a background task so it doesn't block the current goal
            asyncio.create_task(self.execute_goal(folding_goal))

        # 1. Classify Goal
        goal_type = await self._classify_goal(goal)
        print(f"Supervisor classified goal as: {goal_type}")

        final_result = ""
        if goal_type == "Open-Ended":
            # 2a. Delegate to UnifiedReasoningAgent for Open-Ended goals
            print("--- Delegating to UnifiedReasoningAgent for Open-Ended goal. ---")
            unified_reasoning_agent = UnifiedReasoningAgent(self.memory_manager)
            final_result = await unified_reasoning_agent.execute_task({"goal": goal})
        else:
            # 2b. Use GoalDecomposer for hierarchical decomposition (Story 1.1)
            print("--- Decomposing goal into hierarchical task tree... ---")
            try:
                goal_tree = await self.goal_decomposer.decompose(goal)
                print(f"Goal tree created with {len(goal_tree.get_all_leaf_tasks())} leaf tasks")
                print(f"Task tree structure:\n{goal_tree.to_json()}")
                
                # 3. Execute the tree using depth-first traversal
                final_result = await self._execute_tree_node(goal_tree.root)
                
            except Exception as e:
                error_msg = f"Goal decomposition failed: {e}"
                print(error_msg)
                return {"status": "failure", "result": error_msg}

        print(f"\n--- Supervisor finished goal: {goal} ---")
        print(f"Final Result: {final_result}")
        return final_result
    
    async def _execute_tree_node(self, node: DecomposedTask, context: Dict = None) -> Any:
        """
        Recursively executes a task tree node using depth-first traversal.
        Implements Story 1.1: Tree execution with parallel/sequential modes.
        
        Args:
            node: The DecomposedTask node to execute
            context: Accumulated context from previous executions
            
        Returns:
            Result of executing this node and its children
        """
        context = context or {}
        node.status = TaskStatus.IN_PROGRESS
        
        print(f"\n--- Executing task [{node.task_id}] (depth={node.depth}): {node.task} ---")
        
        # Emit dispatch event
        dispatch_payload = {'event_type': 'task_dispatch', 'task_id': node.task_id, 'task': node.task}
        asyncio.create_task(self.metacognition_agent.execute_task(dispatch_payload))
        
        try:
            if node.children:
                # Execute children based on execution mode
                if node.execution_mode == ExecutionMode.PARALLEL:
                    # Execute all children in parallel
                    child_tasks = [
                        self._execute_tree_node(child, context.copy())
                        for child in node.children
                    ]
                    results = await asyncio.gather(*child_tasks, return_exceptions=True)
                    
                    # Check for exceptions and handle them
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            await self._handle_task_failure(node.children[i], str(result), context)
                        else:
                            context[f"child_{i}_result"] = result
                    
                    node.status = TaskStatus.COMPLETED
                    self.goal_decomposer.mark_task_completed(node)
                    return {"status": "success", "children_results": results}
                    
                else:
                    # Execute children sequentially (depth-first)
                    results = []
                    for i, child in enumerate(node.children):
                        result = await self._execute_tree_node(child, context)
                        
                        # Handle failure with potential refinement (Story 1.2)
                        if isinstance(result, dict) and result.get("status") == "failure":
                            refined = await self._handle_task_failure(child, result.get("result"), context)
                            if refined:
                                # Retry with refined subtask
                                result = await self._execute_tree_node(child, context)
                        
                        results.append(result)
                        context[f"step_{i+1}_result"] = result
                    
                    node.status = TaskStatus.COMPLETED
                    self.goal_decomposer.mark_task_completed(node)
                    return results[-1] if results else {"status": "success"}
            
            else:
                # This is a leaf node - determine how to execute it
                result = await self._execute_leaf_task(node, context)
                
                # Handle failure with potential refinement (Story 1.2)
                if isinstance(result, dict) and result.get("status") == "failure":
                    refined = await self._handle_task_failure(node, result.get("result"), context)
                    if refined and node.children:
                        # Refinement added children, execute them
                        result = await self._execute_tree_node(node, context)
                    else:
                        node.mark_failed(result.get("result"))
                        return result
                
                node.status = TaskStatus.COMPLETED
                self.goal_decomposer.mark_task_completed(node)
                return result
                
        except Exception as e:
            error_msg = f"Task execution failed: {e}"
            node.mark_failed(error_msg)
            print(error_msg)
            return {"status": "failure", "result": error_msg}
    
    async def _execute_leaf_task(self, node: DecomposedTask, context: Dict) -> Any:
        """
        Executes a leaf task by mapping it to an appropriate agent or tool.
        
        Args:
            node: The leaf DecomposedTask to execute
            context: Accumulated execution context
            
        Returns:
            Result of task execution
        """
        # Map task types to specialist agents
        agent_mapping = {
            "analysis": "AnalystAgent",
            "research": "AnalystAgent",
            "implementation": None,  # Use tool registry directly
            "verification": None,
        }
        
        agent_name = agent_mapping.get(node.task_type.value)
        
        if agent_name and agent_name in self.specialist_registry:
            # Execute via specialist agent
            agent_class = self.specialist_registry[agent_name]
            agent_instance = agent_class(self.memory_manager) if hasattr(agent_class, '__init__') and 'memory_manager' in str(agent_class.__init__.__code__.co_varnames) else agent_class()
            result = await agent_instance.execute_task({
                "task": node.task,
                "task_type": node.task_type.value,
                "context": context
            })
            return result
        else:
            # For implementation/verification, try to find appropriate tool
            # or return a placeholder indicating the task needs manual handling
            return {
                "status": "success",
                "result": f"Leaf task '{node.task}' acknowledged. Type: {node.task_type.value}",
                "needs_manual_execution": True
            }
    
    async def _handle_task_failure(self, node: DecomposedTask, error_msg: str, context: Dict) -> bool:
        """
        Handles task failure by checking for ambiguity and triggering refinement.
        Implements Story 1.2: Dynamic Subtask Refinement.
        
        Args:
            node: The failed task
            error_msg: Error message from the failure
            context: Execution context
            
        Returns:
            True if task was refined and should be retried
        """
        # Check for ambiguity indicators
        ambiguity_indicators = [
            "too vague", "unclear", "ambiguous", "not specific",
            "need more details", "undefined", "incomplete"
        ]
        
        is_ambiguous = any(indicator in (error_msg or "").lower() for indicator in ambiguity_indicators)
        
        if is_ambiguous or node.depth < DecomposedTask.MAX_DEPTH - 1:
            print(f"--- Task needs refinement: {node.task} ---")
            print(f"--- Triggering just-in-time re-decomposition (Story 1.2) ---")
            
            # Mark for refinement and trigger re-decomposition
            node.status = TaskStatus.NEEDS_REFINEMENT
            await self.goal_decomposer.refine_subtask(node)
            
            if node.children:
                print(f"--- Refinement successful: {len(node.children)} new subtasks created ---")
                return True
        
        return False

