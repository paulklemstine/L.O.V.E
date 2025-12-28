"""
Tool Execution Node for the DeepAgent graph (The Sandbox).

This node is responsible for:
1. Parsing tool call requests from the previous AI message
2. Looking up tools in the ToolRegistry
3. Executing tools with provided arguments
4. Wrapping results in ToolMessage objects
5. Handling errors gracefully without crashing the main loop
"""
import asyncio
import traceback
from typing import Dict, Any, List
from core.state import DeepAgentState
from langchain_core.messages import ToolMessage
import core.logging

# Import tools for direct lookup (fallback)
from core.tools import (
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve,
    speak_to_creator
)

# Fallback tool map for backward compatibility
FALLBACK_TOOL_MAP = {
    "execute": execute,
    "decompose_and_solve_subgoal": decompose_and_solve_subgoal,
    "evolve": evolve,
    "post_to_bluesky": post_to_bluesky,
    "read_file": read_file,
    "write_file": write_file,
    "scan_network": scan_network,
    "probe_target": probe_target,
    "perform_webrequest": perform_webrequest,
    "analyze_json_file": analyze_json_file,
    "research_and_evolve": research_and_evolve,
    "speak_to_creator": speak_to_creator
}

# Error message template for standardized error responses
ERROR_TEMPLATE = "Error executing tool '{tool_name}': {error_details}. Try again with different parameters."


def _get_tool_from_registry(tool_name: str):
    """
    Attempts to get a tool from the global ToolRegistry.
    Falls back to FALLBACK_TOOL_MAP if registry is not available.
    """
    try:
        from core.tool_registry import get_global_registry
        registry = get_global_registry()
        if tool_name in registry:
            return registry.get_tool(tool_name)
    except ImportError:
        pass
    except KeyError:
        pass
    
    # Try fallback map
    return FALLBACK_TOOL_MAP.get(tool_name)


async def _safe_execute_tool(tool_func, tool_args: Dict[str, Any]) -> str:
    """
    Safely executes a tool function with the provided arguments.
    
    Handles both sync and async tools, and catches all exceptions
    to return error strings instead of crashing.
    
    Returns:
        The tool result as a string, or an error message.
    """
    try:
        # Check if tool is a LangChain tool with ainvoke/invoke methods
        if hasattr(tool_func, "ainvoke"):
            result = await tool_func.ainvoke(tool_args)
        elif hasattr(tool_func, "invoke"):
            result = tool_func.invoke(tool_args)
        # Check if it's a coroutine function
        elif asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        # Regular sync function
        else:
            result = tool_func(**tool_args)
        
        return str(result)
    
    except TypeError as e:
        # Usually indicates wrong arguments
        return f"Argument error: {e}. Check parameter names and types."
    except Exception as e:
        # Catch-all for any other errors
        error_trace = traceback.format_exc()
        core.logging.log_event(
            f"Tool execution error: {e}\n{error_trace}",
            "WARNING"
        )
        return f"Execution failed: {e}"


async def tool_execution_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Executes tools requested by the LLM in the previous message.
    
    This node:
    1. Parses tool_calls from the last AI message
    2. Looks up each tool in the ToolRegistry (with fallback)
    3. Executes each tool with provided arguments
    4. Wraps results in ToolMessage objects
    5. Increments loop_count for the self-correction guardrail
    6. Never crashes - returns error strings on failure
    
    Returns:
        State update with ToolMessage outputs and incremented loop_count
    """
    last_message = state["messages"][-1]
    outputs = []
    current_loop_count = state.get("loop_count", 0)
    
    # Check if there are tool calls to process
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        core.logging.log_event(
            "Tool execution node called but no tool_calls found in last message.",
            "WARNING"
        )
        return {"messages": [], "loop_count": current_loop_count}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", f"call_{tool_name}")
        
        core.logging.log_event(
            f"Executing tool '{tool_name}' with args: {tool_args}",
            "INFO"
        )
        
        # Look up the tool
        tool_func = _get_tool_from_registry(tool_name)
        
        if tool_func is None:
            # Tool not found
            error_msg = ERROR_TEMPLATE.format(
                tool_name=tool_name,
                error_details=f"Tool '{tool_name}' not found in registry"
            )
            core.logging.log_event(error_msg, "WARNING")
            result = error_msg
        else:
            # Execute the tool safely
            result = await _safe_execute_tool(tool_func, tool_args)
        
        # Create ToolMessage with the result
        tool_message = ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        )
        outputs.append(tool_message)
        
        core.logging.log_event(
            f"Tool '{tool_name}' completed. Result length: {len(result)} chars",
            "DEBUG"
        )
    
    # Increment loop count for self-correction guardrail
    new_loop_count = current_loop_count + 1
    
    return {
        "messages": outputs,
        "loop_count": new_loop_count
    }

