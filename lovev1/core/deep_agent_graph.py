"""
DeepAgent Protocol - Story 1.2: The Recursive Graph Engine

LangGraph-based cyclic graph architecture implementing:
Plan → Execute → Critic → (Decide: Finalize or Replan)

This module provides the graph builder and node functions for the
DeepAgent Protocol, enabling recursive self-correction.
"""

import logging
from typing import Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END
from core.state import DeepAgentState, create_initial_state
from core.agents.planner_agent import create_plan
from core.agents.critic_agent import critique_execution, decide_next_node
from core.tool_registry import ToolResult
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from core.logging import log_event

logger = logging.getLogger(__name__)

# Circuit breaker for the entire graph execution
graph_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)


# =============================================================================
# EXECUTOR NODE
# =============================================================================

async def execute_step(state: DeepAgentState) -> DeepAgentState:
    """
    Executor node: Executes the current step from the plan.
    
    Takes the next unexecuted step from the plan and attempts to
    execute it using available tools or direct generation.
    """
    from core.llm_api import run_llm
    from core.tool_registry import get_global_registry, ToolResult
    
    plan = state.get("plan", [])
    past_steps = list(state.get("past_steps", []))
    
    # Find the next step to execute
    executed_count = len(past_steps)
    
    if executed_count >= len(plan):
        # All steps executed
        log_event("ExecutorNode: All steps completed", "INFO")
        return state
    
    current_step = plan[executed_count]
    log_event(f"ExecutorNode: Executing step {executed_count + 1}/{len(plan)}: {current_step}", "INFO")
    
    # Check if this is an image generation step (Story 3.3)
    if _is_image_step(current_step):
        result = await _execute_image_step(current_step, state)
    else:
        result = await _execute_general_step(current_step, state)
    
    # Record the step result
    action_taken = result.get("action", "direct_execution")
    outcome = result.get("observation", str(result.get("data", "")))
    
    past_steps.append((current_step, action_taken, outcome))
    state["past_steps"] = past_steps
    
    return state


def _is_image_step(step: str) -> bool:
    """Check if a step involves image generation."""
    image_keywords = ["generate image", "create image", "produce image", "make image", 
                      "image generation", "visual", "artwork", "picture"]
    step_lower = step.lower()
    return any(keyword in step_lower for keyword in image_keywords)


async def _execute_image_step(step: str, state: DeepAgentState) -> Dict[str, Any]:
    """
    Execute an image generation step with Visual Director handshake.
    
    Story 3.3: Consults VisualDirector before calling ImageAPI.
    """
    try:
        from core.visual_director import VisualDirector
        from core.image_api import generate_image
        
        director = VisualDirector()
        
        # Get visual specification
        visual_spec = await director.direct_scene(step)
        
        # Synthesize the image prompt
        subliminal = state.get("input", "")[:50]  # Use part of original input
        image_prompt = director.synthesize_image_prompt(visual_spec, subliminal)
        
        # Generate the image
        image = await generate_image(image_prompt)
        
        return {
            "status": "success",
            "action": "generate_image_with_director",
            "data": {"image": image, "prompt": image_prompt},
            "observation": f"Generated image using Visual Director. Prompt: {image_prompt[:100]}..."
        }
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {
            "status": "error",
            "action": "generate_image",
            "data": {"error": str(e)},
            "observation": f"Image generation failed: {e}"
        }


async def _execute_general_step(step: str, state: DeepAgentState) -> Dict[str, Any]:
    """
    Execute a general step using LLM and tools.
    
    Story 2.3: Uses dynamic tool retrieval to select only relevant tools
    for the current step, optimizing context window usage.
    """
    from core.llm_api import run_llm
    from core.tool_registry import get_global_registry
    from core.tool_retriever import format_tools_for_step, retrieve_tools_for_step
    
    # Get scratchpad context for corrections
    scratchpad = state.get("scratchpad", "")
    original_input = state.get("input", "")
    
    # Story 2.3: Dynamic tool retrieval - only get relevant tools
    registry = get_global_registry()
    relevant_tools_text = format_tools_for_step(step, registry)
    relevant_tools = retrieve_tools_for_step(step, registry, max_tools=5)
    
    # Store retrieved tools in state for reference
    state["retrieved_tools"] = [t.name for t in relevant_tools]
    state["tool_query"] = step
    
    # Build execution prompt with ONLY relevant tools
    execution_prompt = f"""Execute the following step:
STEP: {step}

ORIGINAL REQUEST: {original_input}

{f"CORRECTIONS FROM PREVIOUS ATTEMPT: {scratchpad}" if scratchpad else ""}

{relevant_tools_text}

If you need to use one of the above tools, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Otherwise, provide your response directly.
"""

    try:
        result = await run_llm(execution_prompt, purpose="execution")
        response_text = result.get("result", "")
        
        # Check if response contains a tool call
        if "<tool_call>" in response_text:
            tool_result = await _parse_and_execute_tool(response_text, state)
            return tool_result
        else:
            return {
                "status": "success",
                "action": "direct_response",
                "data": response_text,
                "observation": response_text[:500]
            }
    except Exception as e:
        logger.error(f"Step execution failed: {e}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"error": str(e)},
            "observation": f"Execution failed: {e}"
        }


async def _parse_and_execute_tool(response_text: str, state: DeepAgentState) -> Dict[str, Any]:
    """Parse and execute a tool call from the response."""
    import json
    import re
    
    try:
        # Extract tool call JSON
        match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', response_text, re.DOTALL)
        if not match:
            return {
                "status": "error",
                "action": "parse_failed",
                "data": {},
                "observation": "Failed to parse tool call from response"
            }
        
        tool_call = json.loads(match.group(1))
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        
        # Get the tool from registry
        from core.tool_registry import get_global_registry
        registry = get_global_registry()
        
        if tool_name not in registry:
            return {
                "status": "error",
                "action": f"tool_not_found:{tool_name}",
                "data": {},
                "observation": f"Tool '{tool_name}' not found in registry"
            }
        
        tool_func = registry.get_tool(tool_name)
        
        # Execute the tool
        import asyncio
        import inspect
        
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        else:
            result = tool_func(**tool_args)
        
        # Handle ToolResult type
        if isinstance(result, ToolResult):
            return {
                "status": result.status,
                "action": f"tool:{tool_name}",
                "data": result.data,
                "observation": result.observation
            }
        else:
            return {
                "status": "success",
                "action": f"tool:{tool_name}",
                "data": result,
                "observation": str(result)[:500]
            }
            
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "action": "json_parse_failed",
            "data": {"error": str(e)},
            "observation": f"Failed to parse tool call JSON: {e}"
        }
    except Exception as e:
        return {
            "status": "error",
            "action": "tool_execution_failed",
            "data": {"error": str(e)},
            "observation": f"Tool execution failed: {e}"
        }


# =============================================================================
# FINALIZE NODE
# =============================================================================

async def finalize_session(state: DeepAgentState) -> DeepAgentState:
    """
    Finalize node: Wraps up the session and triggers memory folding.
    
    Story 3.2: Passes scratchpad and chat_history to Memory Folding Agent.
    """
    log_event("FinalizeNode: Finalizing session", "INFO")
    
    # Trigger memory folding if we have content
    scratchpad = state.get("scratchpad", "")
    past_steps = state.get("past_steps", [])
    
    if scratchpad or past_steps:
        try:
            from core.agents.memory_folding_agent import MemoryFoldingAgent
            
            folder = MemoryFoldingAgent()
            
            # Prepare content for folding
            content_to_fold = f"Input: {state.get('input', '')}\n\n"
            content_to_fold += f"Plan: {state.get('plan', [])}\n\n"
            content_to_fold += f"Steps Executed:\n"
            for step, action, result in past_steps:
                content_to_fold += f"  - {step}: {action} -> {result[:100]}...\n"
            content_to_fold += f"\nScratchpad:\n{scratchpad}\n"
            content_to_fold += f"\nCriticism:\n{state.get('criticism', '')}"
            
            # Fold the content (async if available)
            if hasattr(folder, 'fold_async'):
                await folder.fold_async(content_to_fold)
            elif hasattr(folder, 'fold'):
                folder.fold(content_to_fold)
            
            log_event("FinalizeNode: Memory folding completed", "INFO")
            
        except Exception as e:
            log_event(f"FinalizeNode: Memory folding failed: {e}", "WARNING")
    
    # Set stop reason if not already set
    if not state.get("stop_reason"):
        state["stop_reason"] = "completed"
    
    return state


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_deep_agent_graph() -> StateGraph:
    """
    Build the DeepAgent LangGraph.
    
    Story 1.2: Creates the cyclic graph:
    Planner → Executor → Critic → (Finalize | Planner)
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state schema
    graph = StateGraph(DeepAgentState)
    
    # Add nodes
    graph.add_node("planner", create_plan)
    graph.add_node("executor", execute_step)
    graph.add_node("critic", critique_execution)
    graph.add_node("finalize", finalize_session)
    
    # Add edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    
    # Conditional edge from critic
    graph.add_conditional_edges(
        "critic",
        decide_next_node,
        {
            "planner": "planner",
            "finalize": "finalize"
        }
    )
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


class DeepAgentGraphRunner:
    """
    High-level runner for the DeepAgent graph.
    
    Provides a simple interface for executing prompts through
    the recursive reasoning loop.
    """
    
    def __init__(self, memory_manager=None, max_loops: int = 5):
        """
        Initialize the graph runner.
        
        Args:
            memory_manager: Optional memory manager for context
            max_loops: Maximum Plan→Execute→Critic cycles
        """
        self.memory_manager = memory_manager
        self.max_loops = max_loops
        self.graph = build_deep_agent_graph()
    
    async def run(self, prompt: str) -> Dict[str, Any]:
        """
        Execute a prompt through the DeepAgent graph.
        
        Args:
            prompt: User query/prompt to process
            
        Returns:
            Final state with results
        """
        # Create initial state
        state = create_initial_state(
            user_input=prompt,
            memory_manager=self.memory_manager,
            max_loops=self.max_loops
        )
        
        log_event(f"DeepAgentGraph: Starting execution for: {prompt[:100]}...", "INFO")
        
        try:
            # Use circuit breaker for protection
            async def execute_graph():
                return await self.graph.ainvoke(state)
            
            final_state = graph_breaker.call(
                lambda: execute_graph()
            )
            
            # If circuit breaker wraps an async call, we need to await it
            if hasattr(final_state, '__await__'):
                final_state = await final_state
            
            log_event(f"DeepAgentGraph: Completed. Stop reason: {final_state.get('stop_reason')}", "INFO")
            
            return {
                "success": True,
                "stop_reason": final_state.get("stop_reason", "completed"),
                "plan": final_state.get("plan", []),
                "past_steps": final_state.get("past_steps", []),
                "criticism": final_state.get("criticism", ""),
                "loops": final_state.get("current_loop", 0)
            }
            
        except CircuitBreakerOpenException as e:
            log_event(f"DeepAgentGraph: Circuit breaker open: {e}", "ERROR")
            return {
                "success": False,
                "error": "Circuit breaker open - system is recovering",
                "stop_reason": "circuit_breaker"
            }
        except Exception as e:
            log_event(f"DeepAgentGraph: Execution failed: {e}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "stop_reason": "error"
            }
    
    def get_graph_diagram(self) -> str:
        """
        Get a Mermaid diagram of the graph.
        
        Returns:
            Mermaid diagram string
        """
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            # Fallback static diagram
            return """
graph TD
    __start__ --> planner
    planner --> executor
    executor --> critic
    critic -->|approved| finalize
    critic -->|rejected| planner
    finalize --> __end__
"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_deep_agent(prompt: str, memory_manager=None, max_loops: int = 5) -> Dict[str, Any]:
    """
    Convenience function to run a prompt through the DeepAgent graph.
    
    Args:
        prompt: User query
        memory_manager: Optional memory manager
        max_loops: Max recursion depth
        
    Returns:
        Execution results
    """
    runner = DeepAgentGraphRunner(memory_manager=memory_manager, max_loops=max_loops)
    return await runner.run(prompt)
