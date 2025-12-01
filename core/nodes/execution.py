from typing import Dict, Any, List
from core.state import DeepAgentState
from langchain_core.messages import ToolMessage
from core.tools import (
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve
)

# Map tool names to functions
TOOL_MAP = {
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
    "research_and_evolve": research_and_evolve
}

async def tool_execution_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Executes tools requested by the LLM.
    """
    last_message = state["messages"][-1]
    outputs = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                try:
                    # Check if tool is async
                    if hasattr(tool_func, "coroutine") or asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func.invoke(tool_args)
                    else:
                        result = tool_func.invoke(tool_args)
                except Exception as e:
                    result = f"Error executing tool {tool_name}: {e}"
            else:
                result = f"Error: Tool {tool_name} not found."
                
            outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name))
            
    return {"messages": outputs}

import asyncio
