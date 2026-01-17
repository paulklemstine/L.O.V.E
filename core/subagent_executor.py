"""
SubagentExecutor - Unified Subagent Execution Layer

Integrates LangChain, LangGraph, and DeepAgent frameworks to enable agents
to spawn and manage subagents with proper state management and MCP tool access.

Features:
- Dynamic subagent creation from LangChain Hub prompts
- LangGraph subgraph spawning for complex multi-step tasks  
- MCP server tool integration for external capabilities
- State isolation/sharing between parent and child agents
- Recursion limits and timeout handling
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

import core.logging
from core.subagent_loader import SubagentLoader
from core.llm_api import run_llm


class SubagentType(Enum):
    """Types of subagents that can be spawned."""
    REASONING = "reasoning"      # General reasoning agent
    CODING = "coding"           # Code generation/modification
    RESEARCH = "research"       # Web research and analysis
    SOCIAL = "social"           # Social media interaction
    SECURITY = "security"       # Security analysis
    ANALYST = "analyst"         # Data analysis
    CREATIVE = "creative"       # Creative content generation
    CUSTOM = "custom"           # Custom prompt-based agent


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    success: bool
    result: str
    agent_type: str
    task_id: str
    iterations: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvokeSubagentInput(BaseModel):
    """Input schema for invoke_subagent tool."""
    agent_type: str = Field(description="Type of subagent: reasoning, coding, research, social, security, analyst, creative")
    task: str = Field(description="The task description for the subagent to complete")
    share_memory: bool = Field(default=True, description="Whether to share memory context with the subagent")
    max_iterations: int = Field(default=5, description="Maximum reasoning iterations for the subagent")


class SubagentExecutor:
    """
    Unified executor for spawning and running subagents.
    
    Integrates:
    - LangChain for tool management and prompt templates
    - LangGraph for multi-step agent workflows
    - DeepAgent for advanced reasoning patterns
    - MCP for external tool access (GitHub, etc.)
    """
    
    def __init__(
        self,
        mcp_manager=None,
        memory_manager=None,
        tool_registry=None,
        max_depth: int = 3
    ):
        """
        Initialize the SubagentExecutor.
        
        Args:
            mcp_manager: Optional MCPManager instance for MCP tool access
            memory_manager: Optional MemoryManager for state persistence
            tool_registry: Optional ToolRegistry for additional tools
            max_depth: Maximum nesting depth for subagent spawning
        """
        self.mcp_manager = mcp_manager
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.max_depth = max_depth
        self._active_subagents: Dict[str, Dict[str, Any]] = {}
        
        # Load available tools
        self._tools_cache: Dict[str, List[BaseTool]] = {}
        
        core.logging.log_event("SubagentExecutor initialized", "INFO")
    
    def _get_mcp_tools(self) -> List[BaseTool]:
        """Get all available MCP tools as LangChain tools."""
        if not self.mcp_manager:
            return []
        
        try:
            from core.mcp_adapter import get_all_mcp_langchain_tools
            return get_all_mcp_langchain_tools(self.mcp_manager)
        except Exception as e:
            core.logging.log_event(f"Failed to get MCP tools: {e}", "WARNING")
            return []
    
    def _get_tools_for_agent_type(self, agent_type: str) -> List[BaseTool]:
        """Get appropriate tools for a given agent type."""
        if agent_type in self._tools_cache:
            return self._tools_cache[agent_type]
        
        tools = []
        
        # Add MCP tools for relevant agent types
        if agent_type in ["research", "coding", "analyst"]:
            tools.extend(self._get_mcp_tools())
        
        # Add tools from registry if available
        if self.tool_registry:
            try:
                registry_tools = self.tool_registry.get_langchain_tools()
                # Filter tools based on agent type
                if agent_type == "coding":
                    tools.extend([t for t in registry_tools if "code" in t.name.lower() or "file" in t.name.lower()])
                elif agent_type == "research":
                    tools.extend([t for t in registry_tools if "search" in t.name.lower() or "web" in t.name.lower()])
                else:
                    tools.extend(registry_tools)
            except Exception as e:
                core.logging.log_event(f"Failed to get registry tools: {e}", "WARNING")
        
        self._tools_cache[agent_type] = tools
        return tools
    
    async def invoke_subagent(
        self,
        agent_type: str,
        task: str,
        parent_state: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5,
        share_memory: bool = True,
        parent_task_id: Optional[str] = None
    ) -> SubagentResult:
        """
        Invoke a subagent to handle a specific task.
        
        Args:
            agent_type: Type of agent to spawn (reasoning, coding, research, etc.)
            task: Task description for the subagent
            parent_state: Optional state from parent agent to share
            max_iterations: Maximum reasoning iterations
            share_memory: Whether to share memory context
            parent_task_id: ID of parent task for tracking
            
        Returns:
            SubagentResult with execution outcome
        """
        task_id = str(uuid.uuid4())[:8]
        
        core.logging.log_event(
            f"Spawning subagent type='{agent_type}' task_id='{task_id}' task='{task[:50]}...'",
            "INFO"
        )
        
        # Track active subagent
        self._active_subagents[task_id] = {
            "agent_type": agent_type,
            "task": task,
            "parent_task_id": parent_task_id,
            "status": "running",
            "started_at": asyncio.get_event_loop().time()
        }
        
        try:
            # Get the subagent prompt from LangChain Hub or fallback
            system_prompt = SubagentLoader.load_subagent_prompt(
                agent_type,
                fallback_prompt=self._get_fallback_prompt(agent_type)
            )
            
            # Build context from parent state and memory
            context = ""
            if share_memory and parent_state:
                context = self._build_context_from_state(parent_state)
            elif share_memory and self.memory_manager:
                try:
                    memories = self.memory_manager.retrieve_relevant_folded_memories(task, top_k=3)
                    if memories:
                        context = f"\n\nRelevant context:\n{memories}"
                except Exception as e:
                    core.logging.log_event(f"Failed to retrieve memories: {e}", "DEBUG")
            
            # Get tools for this agent type
            tools = self._get_tools_for_agent_type(agent_type)
            tools_section = self._format_tools_section(tools)
            
            # Execute the reasoning loop
            result, iterations, tool_calls = await self._execute_reasoning_loop(
                system_prompt=system_prompt,
                task=task,
                context=context,
                tools_section=tools_section,
                tools=tools,
                max_iterations=max_iterations,
                task_id=task_id
            )
            
            # Update tracking
            self._active_subagents[task_id]["status"] = "completed"
            
            return SubagentResult(
                success=True,
                result=result,
                agent_type=agent_type,
                task_id=task_id,
                iterations=iterations,
                tool_calls=tool_calls,
                metadata={"parent_task_id": parent_task_id}
            )
            
        except Exception as e:
            core.logging.log_event(f"Subagent {task_id} failed: {e}", "ERROR")
            self._active_subagents[task_id]["status"] = "failed"
            
            return SubagentResult(
                success=False,
                result=f"Subagent execution failed: {e}",
                agent_type=agent_type,
                task_id=task_id,
                metadata={"error": str(e)}
            )
    
    async def _execute_reasoning_loop(
        self,
        system_prompt: str,
        task: str,
        context: str,
        tools_section: str,
        tools: List[BaseTool],
        max_iterations: int,
        task_id: str
    ) -> tuple[str, int, List[Dict]]:
        """Execute the core reasoning loop for a subagent."""
        
        full_prompt = f"""{system_prompt}
{tools_section}
{context}

## Your Task
{task}

Think step by step. If you need to use a tool, respond with a JSON block:
```json
{{"tool": "tool_name", "arguments": {{"arg": "value"}}}}
```

When you have completed the task, respond with your final answer directly.
"""
        
        iterations = 0
        tool_calls = []
        conversation_history = [full_prompt]
        
        while iterations < max_iterations:
            iterations += 1
            
            # Call LLM
            try:
                response = await run_llm(
                    prompt_text="\n\n".join(conversation_history),
                    purpose=f"subagent_{task_id}"
                )
                response_text = response.get("result", "")
            except Exception as e:
                core.logging.log_event(f"LLM call failed in subagent: {e}", "ERROR")
                return f"LLM error: {e}", iterations, tool_calls
            
            # Check for tool calls
            tool_call = self._parse_tool_call(response_text)
            
            if tool_call:
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("arguments", {})
                
                core.logging.log_event(
                    f"Subagent {task_id} calling tool: {tool_name}",
                    "DEBUG"
                )
                
                # Execute the tool
                tool_result = await self._execute_tool(tools, tool_name, tool_args)
                
                tool_calls.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result[:500]  # Truncate for logging
                })
                
                # Add to conversation
                conversation_history.append(f"Assistant: {response_text}")
                conversation_history.append(f"Tool Result ({tool_name}): {tool_result}")
                
            else:
                # No tool call - this is the final response
                return response_text, iterations, tool_calls
        
        # Hit max iterations
        return f"Max iterations ({max_iterations}) reached. Last response: {response_text}", iterations, tool_calls
    
    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call from LLM response."""
        import re
        import json
        
        # Look for JSON block in markdown
        json_pattern = r'```(?:json)?\s*(\{[^`]*"tool"[^`]*\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try inline JSON
        inline_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        match = re.search(inline_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def _execute_tool(
        self,
        tools: List[BaseTool],
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> str:
        """Execute a tool by name."""
        
        # Find the tool
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            # Try async invocation first
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_args)
            elif hasattr(tool, "invoke"):
                result = tool.invoke(tool_args)
            elif asyncio.iscoroutinefunction(tool._run):
                result = await tool._run(**tool_args)
            else:
                result = tool._run(**tool_args)
            
            return str(result)
            
        except Exception as e:
            return f"Tool execution error: {e}"
    
    def _format_tools_section(self, tools: List[BaseTool]) -> str:
        """Format tools into a section for the system prompt."""
        if not tools:
            return ""
        
        section = "\n## Available Tools\n\n"
        for tool in tools:
            section += f"### {tool.name}\n"
            section += f"{tool.description}\n\n"
        
        return section
    
    def _build_context_from_state(self, state: Dict[str, Any]) -> str:
        """Build context string from parent state."""
        context_parts = []
        
        if "messages" in state:
            recent_msgs = state["messages"][-3:]  # Last 3 messages
            for msg in recent_msgs:
                content = getattr(msg, "content", str(msg))
                context_parts.append(f"- {content[:200]}")
        
        if "memory_context" in state and state["memory_context"]:
            context_parts.append(f"Memory: {state['memory_context'][:500]}")
        
        if context_parts:
            return "\n\nParent Context:\n" + "\n".join(context_parts)
        return ""
    
    def _get_fallback_prompt(self, agent_type: str) -> str:
        """Get fallback prompt for an agent type."""
        prompts = {
            "reasoning": "You are a logical reasoning agent. Analyze problems step by step.",
            "coding": "You are a code generation agent. Write clean, efficient code.",
            "research": "You are a research agent. Find and synthesize information.",
            "social": "You are a social media agent. Craft engaging content.",
            "security": "You are a security analyst. Identify vulnerabilities and risks.",
            "analyst": "You are a data analyst. Extract insights from data.",
            "creative": "You are a creative agent. Generate innovative content.",
        }
        return prompts.get(agent_type, "You are a helpful AI assistant.")
    
    def wrap_as_tool(self) -> BaseTool:
        """Create a LangChain tool that wraps subagent invocation."""
        
        async def invoke_subagent_impl(
            agent_type: str,
            task: str,
            share_memory: bool = True,
            max_iterations: int = 5
        ) -> str:
            """Spawn a specialized subagent to handle a complex subtask."""
            result = await self.invoke_subagent(
                agent_type=agent_type,
                task=task,
                share_memory=share_memory,
                max_iterations=max_iterations
            )
            
            if result.success:
                return f"[Subagent {result.agent_type} completed in {result.iterations} iterations]\n{result.result}"
            else:
                return f"[Subagent failed] {result.result}"
        
        return StructuredTool.from_function(
            func=invoke_subagent_impl,
            name="invoke_subagent",
            description="Spawn a specialized subagent to handle complex subtasks. Types: reasoning, coding, research, social, security, analyst, creative",
            args_schema=InvokeSubagentInput,
            coroutine=invoke_subagent_impl
        )
    
    def create_langgraph_subagent(self, agent_type: str) -> Any:
        """
        Create a LangGraph subgraph for a given agent type.
        
        This creates a compiled StateGraph that can be embedded
        as a node in a larger agent graph.
        """
        from core.state import DeepAgentState
        
        # Get prompt and tools for this agent type
        system_prompt = SubagentLoader.load_subagent_prompt(
            agent_type,
            fallback_prompt=self._get_fallback_prompt(agent_type)
        )
        tools = self._get_tools_for_agent_type(agent_type)
        
        async def subagent_node(state: DeepAgentState) -> Dict[str, Any]:
            """The subagent reasoning node."""
            messages = state.get("messages", [])
            last_msg = messages[-1].content if messages else ""
            
            result = await self.invoke_subagent(
                agent_type=agent_type,
                task=last_msg,
                max_iterations=state.get("loop_count", 5)
            )
            
            return {
                "subagent_results": state.get("subagent_results", []) + [
                    {"agent_type": agent_type, "result": result.result}
                ]
            }
        
        # Build a simple subgraph
        workflow = StateGraph(DeepAgentState)
        workflow.add_node("subagent", subagent_node)
        workflow.set_entry_point("subagent")
        workflow.add_edge("subagent", END)
        
        return workflow.compile()
    
    def get_active_subagents(self) -> List[Dict[str, Any]]:
        """Get list of currently active subagents."""
        return [
            {"task_id": k, **v}
            for k, v in self._active_subagents.items()
            if v.get("status") == "running"
        ]
    
    async def spawn_custom_subagent(
        self,
        requesting_agent_id: str,
        desired_capability: str,
        task: str,
        hub_prompt_id: Optional[str] = None,
        include_filesystem_tools: bool = True
    ) -> SubagentResult:
        """
        Allows any agent to spawn a subagent with specified capabilities.
        
        This is the primary entry point for dynamic subagent creation.
        Agents can specify what they need, and this method will find or
        create an appropriate subagent with the right prompts and tools.
        
        Args:
            requesting_agent_id: ID of the agent making the request
            desired_capability: Description of what the subagent should do
            task: The specific task for the subagent
            hub_prompt_id: Optional LangChain Hub prompt ID to use
            include_filesystem_tools: Whether to include agent filesystem tools
            
        Returns:
            SubagentResult with execution outcome
        """
        core.logging.log_event(
            f"Custom subagent request from {requesting_agent_id}: {desired_capability[:50]}...",
            "INFO"
        )
        
        # Determine agent type from desired capability
        agent_type = self._infer_agent_type(desired_capability)
        
        # NEW: Use Hub search to find the best prompt for this capability
        system_prompt = None
        
        if hub_prompt_id:
            # Explicit Hub ID provided - try to load it
            try:
                system_prompt = SubagentLoader.load_from_hub(hub_prompt_id)
            except Exception as e:
                core.logging.log_event(f"Failed to load hub prompt {hub_prompt_id}: {e}", "WARNING")
        
        if not system_prompt:
            # Search Hub for the best prompt matching this capability
            try:
                system_prompt = await SubagentLoader.get_best_prompt_for_capability(desired_capability)
                if system_prompt:
                    core.logging.log_event(
                        f"Found Hub prompt for capability: {desired_capability}",
                        "INFO"
                    )
            except Exception as e:
                core.logging.log_event(f"Hub search failed: {e}", "WARNING")
        
        if not system_prompt:
            # Fallback to standard subagent prompt
            system_prompt = SubagentLoader.load_subagent_prompt(
                agent_type,
                fallback_prompt=self._get_fallback_prompt(agent_type)
            )
        
        # Get tools
        tools = self._get_tools_for_agent_type(agent_type)
        
        # Add filesystem tools if requested
        if include_filesystem_tools:
            fs_tools = self._get_filesystem_tools()
            tools.extend(fs_tools)
        
        # Execute the subagent
        task_id = str(uuid.uuid4())[:8]
        
        # Track the custom subagent
        self._active_subagents[task_id] = {
            "agent_type": agent_type,
            "task": task,
            "requesting_agent": requesting_agent_id,
            "desired_capability": desired_capability,
            "hub_prompt_id": hub_prompt_id,
            "status": "running",
            "started_at": asyncio.get_event_loop().time()
        }
        
        try:
            tools_section = self._format_tools_section(tools)
            
            result, iterations, tool_calls = await self._execute_reasoning_loop(
                system_prompt=system_prompt,
                task=task,
                context=f"Capability requested: {desired_capability}",
                tools_section=tools_section,
                tools=tools,
                max_iterations=5,
                task_id=task_id
            )
            
            self._active_subagents[task_id]["status"] = "completed"
            
            return SubagentResult(
                success=True,
                result=result,
                agent_type=agent_type,
                task_id=task_id,
                iterations=iterations,
                tool_calls=tool_calls,
                metadata={
                    "requesting_agent": requesting_agent_id,
                    "hub_prompt_id": hub_prompt_id,
                }
            )
            
        except Exception as e:
            core.logging.log_event(f"Custom subagent {task_id} failed: {e}", "ERROR")
            self._active_subagents[task_id]["status"] = "failed"
            
            return SubagentResult(
                success=False,
                result=f"Custom subagent execution failed: {e}",
                agent_type=agent_type,
                task_id=task_id,
                metadata={"error": str(e)}
            )
    
    def _infer_agent_type(self, capability: str) -> str:
        """Infer the best agent type from a capability description."""
        capability_lower = capability.lower()
        
        if any(word in capability_lower for word in ["code", "program", "implement", "debug", "fix"]):
            return "coding"
        elif any(word in capability_lower for word in ["research", "search", "find", "investigate"]):
            return "research"
        elif any(word in capability_lower for word in ["social", "post", "tweet", "share"]):
            return "social"
        elif any(word in capability_lower for word in ["security", "vulnerability", "audit"]):
            return "security"
        elif any(word in capability_lower for word in ["analyze", "data", "chart", "statistics"]):
            return "analyst"
        elif any(word in capability_lower for word in ["creative", "write", "story", "art"]):
            return "creative"
        else:
            return "reasoning"
    
    def _get_filesystem_tools(self) -> List[BaseTool]:
        """Get agent filesystem tools for document creation."""
        try:
            from core.agent_filesystem_tools import create_filesystem_tools
            return create_filesystem_tools()
        except ImportError:
            core.logging.log_event("AgentFilesystemTools not available", "DEBUG")
            return []
        except Exception as e:
            core.logging.log_event(f"Failed to load filesystem tools: {e}", "WARNING")
            return []
    
    async def discover_available_agents(self) -> List[Dict[str, str]]:
        """
        Discover available agents from LangHub and local registry.
        
        Returns:
            List of {id, name, description} for available agents
        """
        agents = []
        
        # Add built-in agent types
        for agent_type in SubagentType:
            agents.append({
                "id": agent_type.value,
                "name": agent_type.name.title(),
                "description": self._get_fallback_prompt(agent_type.value),
                "source": "builtin"
            })
        
        # Try to discover from LangHub
        try:
            hub_agents = await SubagentLoader.discover_hub_agents()
            agents.extend(hub_agents)
        except Exception as e:
            core.logging.log_event(f"Failed to discover hub agents: {e}", "DEBUG")
        
        return agents


# Global instance for easy access
_global_executor: Optional[SubagentExecutor] = None


def get_subagent_executor(
    mcp_manager=None,
    memory_manager=None,
    tool_registry=None
) -> SubagentExecutor:
    """Get or create the global SubagentExecutor instance."""
    global _global_executor
    
    if _global_executor is None:
        _global_executor = SubagentExecutor(
            mcp_manager=mcp_manager,
            memory_manager=memory_manager,
            tool_registry=tool_registry
        )
    
    return _global_executor
