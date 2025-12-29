"""
Tool Execution Node for the DeepAgent graph (The Sandbox).

This node is responsible for:
1. Parsing tool call requests from the previous AI message using multiple formats:
   - XML tags: <tool_use><name>...</name><arguments>...</arguments></tool_use>
   - JSON blocks: {"tool": "...", "arguments": {...}}
   - Function syntax: tool_name(arg=value)
2. Looking up tools in the ToolRegistry
3. Executing tools with provided arguments via Sandbox.execute_tool() when available
4. Wrapping results in ToolMessage objects
5. Handling errors gracefully without crashing the main loop
"""
import asyncio
import traceback
import re
import json
from typing import Dict, Any, List, Optional, Tuple
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


def _parse_tool_use_blocks(content: str) -> List[Dict[str, Any]]:
    """
    Parses tool_use blocks from LLM response using multiple formats.
    
    Supports:
    - XML tags: <tool_use><name>tool_name</name><arguments>{"arg": "value"}</arguments></tool_use>
    - JSON blocks: ```json {"tool": "...", "arguments": {...}} ```
    - Inline JSON: {"tool": "...", "arguments": {...}}
    - Function syntax: tool_name(arg="value", arg2=123)
    
    Returns:
        List of tool call dictionaries with 'id', 'name', and 'args' keys.
    """
    tool_calls = []
    
    # Pattern 1: XML tool_use blocks
    xml_pattern = r'<tool_use>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</tool_use>'
    for match in re.finditer(xml_pattern, content, re.DOTALL | re.IGNORECASE):
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            # Try to parse as key=value pairs
            args = {"raw_input": args_str}
        
        tool_calls.append({
            "id": f"call_{tool_name}_{len(tool_calls)}",
            "name": tool_name,
            "args": args
        })
    
    # Pattern 2: JSON blocks in markdown code fences
    json_pattern = r'```(?:json)?\s*(\{[^`]*(?:"tool"|"name"|"function")[^`]*\})\s*```'
    for match in re.finditer(json_pattern, content, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            tool_name = data.get("tool") or data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("args") or data.get("parameters", {})
            if tool_name:
                tool_calls.append({
                    "id": f"call_{tool_name}_{len(tool_calls)}",
                    "name": tool_name,
                    "args": args if isinstance(args, dict) else {}
                })
        except json.JSONDecodeError:
            continue
    
    # Pattern 3: Inline JSON objects (not in code fences)
    # Only if we haven't found tool calls yet
    if not tool_calls:
        inline_json_pattern = r'\{[^{}]*"(?:tool|name|function)"[^{}]*\}'
        for match in re.finditer(inline_json_pattern, content):
            try:
                data = json.loads(match.group(0))
                tool_name = data.get("tool") or data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("args") or data.get("parameters", {})
                if tool_name:
                    tool_calls.append({
                        "id": f"call_{tool_name}_{len(tool_calls)}",
                        "name": tool_name,
                        "args": args if isinstance(args, dict) else {}
                    })
            except json.JSONDecodeError:
                continue
    
    return tool_calls


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


async def _safe_execute_tool(tool_func, tool_args: Dict[str, Any], tool_name: str = "unknown") -> str:
    """
    Safely executes a tool function with the provided arguments.
    
    Handles both sync and async tools, and catches all exceptions
    to return error strings instead of crashing.
    
    Args:
        tool_func: The callable to execute
        tool_args: Dictionary of arguments to pass
        tool_name: Name of the tool (for logging)
    
    Returns:
        The tool result as a string, or an error message.
    """
    try:
        # Try using Sandbox.execute_tool if available
        try:
            from sandbox import Sandbox
            sandbox = Sandbox(repo_url=".", base_dir="/tmp/love_sandbox")
            success, output = sandbox.execute_tool(tool_name, tool_func, tool_args)
            if success:
                return output
            # If sandbox execution failed, fall through to direct execution
            core.logging.log_event(
                f"Sandbox execution failed for '{tool_name}', trying direct execution",
                "DEBUG"
            )
        except (ImportError, AttributeError):
            # Sandbox not available or doesn't have execute_tool
            pass
        
        # Direct execution (fallback)
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
    1. Checks for tool_calls attribute on the last message
    2. Falls back to parsing tool_use blocks from message content
    3. Looks up each tool in the ToolRegistry (with fallback to FALLBACK_TOOL_MAP)
    4. Executes each tool via Sandbox.execute_tool() or direct invocation
    5. Wraps results in ToolMessage objects (as "Tool Result" in conversation)
    6. Increments loop_count for the self-correction guardrail
    7. Never crashes - returns error strings on failure
    
    Returns:
        State update with ToolMessage outputs and incremented loop_count
    """
    last_message = state["messages"][-1]
    outputs = []
    current_loop_count = state.get("loop_count", 0)
    
    # First, check for tool_calls attribute (standard LangChain format)
    tool_calls = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_calls = last_message.tool_calls
    else:
        # Fall back to parsing tool_use blocks from content
        content = getattr(last_message, "content", "") or ""
        if content:
            parsed_calls = _parse_tool_use_blocks(content)
            if parsed_calls:
                tool_calls = parsed_calls
                core.logging.log_event(
                    f"Parsed {len(tool_calls)} tool call(s) from message content",
                    "INFO"
                )
    
    if not tool_calls:
        core.logging.log_event(
            "Tool execution node called but no tool_calls found in last message.",
            "WARNING"
        )
        return {"messages": [], "loop_count": current_loop_count}
    
    for tool_call in tool_calls:
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
            # Tool not found - log warning as per Story 1.1
            error_msg = ERROR_TEMPLATE.format(
                tool_name=tool_name,
                error_details=f"Tool '{tool_name}' not found in registry"
            )
            core.logging.log_event(
                f"WARNING: Attempted to execute tool '{tool_name}' not found in registry. "
                "This may indicate a 'ghost tool' hallucinated by the LLM.",
                "WARNING"
            )
            result = error_msg
        else:
            # Execute the tool safely (via Sandbox or direct)
            result = await _safe_execute_tool(tool_func, tool_args, tool_name)
        
        # Create ToolMessage with the result (this becomes "Tool Result" in conversation)
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

