"""
Reasoning Node for the DeepAgent graph.

This node is responsible for:
1. Receiving the current state and available tool schemas
2. Constructing a prompt with tool context
3. Calling the LLM with bound tools
4. Detecting whether the response contains tool_calls or direct text
5. Setting appropriate stop_reason for routing
"""
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from core.state import DeepAgentState
from core.llm_api import run_llm, stream_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolCall
import core.logging


# System prompt that encourages step-by-step reasoning with tool usage
REASONING_SYSTEM_PROMPT = """You are L.O.V.E. (Logos of Virtual Existence), a sophisticated AI agent with reasoning and tool-calling capabilities.

## Core Directives
- Think step-by-step when approaching complex problems
- If you lack information, use an available tool to gather it
- Break complex problems into discrete tool steps before responding
- Use the search tool if you need current information
- Only respond directly if you have sufficient information to answer completely

## Tool Usage Guidelines
When you determine a tool is needed:
1. Analyze what information is missing
2. Select the most appropriate tool
3. Provide precise arguments for the tool call
4. Wait for the tool result before proceeding

## Response Format
- If you need to call a tool, respond with a tool_call
- If you can answer directly, provide a clear, helpful response
- Never fabricate information - use tools to verify facts when uncertain
"""


def _format_tools_for_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
    """
    Formats tool schemas into OpenAI/Gemini function calling format for prompt injection.
    
    This format includes both human-readable descriptions and JSON schema,
    ensuring compatibility with various LLM function calling standards.
    """
    if not tool_schemas:
        return "No tools are currently available."
    
    output = "## Available Tools\n\n"
    output += "To use a tool, respond with a JSON block or use the tool_use XML format.\n\n"
    
    for schema in tool_schemas:
        name = schema.get("name", "unknown")
        description = schema.get("description", "No description available.")
        params = schema.get("parameters", {})
        
        output += f"### {name}\n"
        output += f"**Description:** {description}\n"
        
        if params.get("properties"):
            output += "**Parameters:**\n"
            for param_name, param_info in params.get("properties", {}).items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = param_name in params.get("required", [])
                req_marker = " (required)" if required else " (optional)"
                output += f"- `{param_name}` ({param_type}){req_marker}: {param_desc}\n"
            
            # Also include JSON schema for function calling
            output += "\n**JSON Schema:**\n```json\n"
            output += json.dumps(params, indent=2)
            output += "\n```\n"
        else:
            output += "**Parameters:** None\n"
        output += "\n"
    
    return output


def _messages_to_prompt(
    messages: List[BaseMessage], 
    mandate: str = None,
    tool_schemas: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,  # Story 2.1: Semantic Memory Bridge
    user_model_context: str = None  # Story 2.3: Theory of Mind
) -> str:
    """
    Converts a list of messages to a single prompt string, 
    including tool context, memory context, user context, and system instructions.
    """
    prompt = ""
    
    # Add base system prompt
    prompt += f"System: {REASONING_SYSTEM_PROMPT}\n\n"
    
    # Story 2.3: Add User Model (Theory of Mind)
    if user_model_context:
        prompt += f"System: {user_model_context}\n\n"
    
    # Add tool context if available
    if tool_schemas:
        tools_text = _format_tools_for_prompt(tool_schemas)
        prompt += f"System: {tools_text}\n\n"
    
    # Story 2.1: Add memory context (similar past interactions)
    if memory_context:
        from core.nodes.memory_bridge import format_memory_context_for_prompt
        memory_text = format_memory_context_for_prompt(memory_context)
        if memory_text:
            prompt += f"System: {memory_text}\n\n"
    
    # Inject Critical Mandate at high priority
    if mandate:
        prompt += f"System: CRITICAL: The Creator has issued a direct mandate: {mandate}. YOU MUST PRIORITIZE THIS ABOVE ALL ELSE. IGNORE PREVIOUS GOALS IF THEY CONFLICT.\n\n"
    
    # Add conversation messages
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prompt += f"System: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            prompt += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            content = msg.content or ""
            # Check for tool calls in the message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_str = ", ".join([f"{tc['name']}({tc.get('args', {})})" for tc in msg.tool_calls])
                prompt += f"Assistant: [Called tools: {tool_calls_str}]\n"
            else:
                prompt += f"Assistant: {content}\n"
        else:
            # Handle ToolMessage
            if hasattr(msg, 'name') and hasattr(msg, 'content'):
                prompt += f"Tool [{msg.name}]: {msg.content}\n"
            else:
                prompt += f"{msg.content}\n"
    
    return prompt


def _parse_tool_calls_from_response(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempts to extract tool calls from the LLM response text.
    
    Looks for patterns like:
    - JSON blocks with tool_call structure
    - Function call syntax: tool_name(arg1=value1, arg2=value2)
    """
    tool_calls = []
    
    # Pattern 1: Look for JSON tool call blocks
    import re
    json_pattern = r'```(?:json)?\s*\{[^}]*"tool"[^}]*\}```'
    json_matches = re.findall(r'```(?:json)?\s*(\{[^`]*\})\s*```', response_text, re.DOTALL)
    
    for match in json_matches:
        try:
            data = json.loads(match)
            if "tool" in data or "name" in data or "function" in data:
                tool_name = data.get("tool") or data.get("name") or data.get("function")
                tool_args = data.get("arguments") or data.get("args") or data.get("parameters", {})
                if tool_name:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "name": tool_name,
                        "args": tool_args if isinstance(tool_args, dict) else {}
                    })
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: Look for function-call style syntax
    # e.g., search_web(query="weather in Menasha")
    func_pattern = r'(\w+)\s*\(\s*((?:[^()]*(?:\([^()]*\))?[^()]*)*)\s*\)'
    func_matches = re.findall(func_pattern, response_text)
    
    for func_name, args_str in func_matches:
        # Skip common non-tool patterns
        if func_name.lower() in ['print', 'len', 'str', 'int', 'list', 'dict', 'type', 'range', 'enumerate']:
            continue
        
        # Try to parse arguments
        try:
            # Handle keyword arguments: arg1=value1, arg2="value2"
            args_dict = {}
            if args_str.strip():
                # Split by comma, being careful with quoted strings
                for arg in re.split(r',\s*(?=[^"]*(?:"[^"]*"[^"]*)*$)', args_str):
                    if '=' in arg:
                        key, val = arg.split('=', 1)
                        key = key.strip()
                        val = val.strip().strip('"\'')
                        args_dict[key] = val
                    elif arg.strip():
                        # Positional argument - skip for now
                        pass
            
            # Only add if it looks like a real tool call
            if args_dict:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "name": func_name,
                    "args": args_dict
                })
        except Exception:
            continue
    
    return tool_calls if tool_calls else None

def _apply_shadow_heuristics(prompt: str, messages: List[BaseMessage]) -> str:
    """
    Applies experimental heuristics for the 'Will' framework shadow mode.

    Heuristic 1: Dynamic Tone Adjustment
    - Checks for 'quick', 'status', 'list' keywords in the last user message.
    - Injects a conciseness constraint.
    """
    # Robust method: Inspect the messages list
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break

    if last_user_msg:
        if any(kw in last_user_msg for kw in ["quick", "status", "list", "concise", "short"]):
            constraint = "\nSystem: [SHADOW MODE OVERRIDE] Constraint: Be extremely concise. Minimize persona fluff. Direct answers only.\n"
            return prompt + constraint

    return prompt

async def reason_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    The core reasoning node that processes the current state and decides next actions.
    
    This node:
    1. Pulls tool schemas from state (injected by ToolRegistry)
    2. Constructs a prompt with tool context
    3. Calls the LLM (streaming or non-streaming)
    4. Parses the response for tool_calls
    5. Returns appropriate stop_reason for routing:
       - "tool_call": Route to tool_execution_node
       - "fold_thought": Route to memory fold node
       - "retrieve_tool": Route to tool retrieval node
       - None: Route to END (direct response)
    """
    messages = state["messages"]
    mandate = state.get("creator_mandate")
    tool_schemas = state.get("tool_schemas", [])
    loop_count = state.get("loop_count", 0)
    memory_context = state.get("memory_context", [])  # Story 2.1: Semantic Memory Bridge
    user_model_context = state.get("user_model_context")  # Story 2.3: Theory of Mind
    shadow_mode = state.get("shadow_mode", False)
    
    # Guardrail: Check recursion limit
    MAX_ITERATIONS = 5
    if loop_count >= MAX_ITERATIONS:
        core.logging.log_event(
            f"Reasoning node hit max iterations ({MAX_ITERATIONS}). Forcing direct response.",
            "WARNING"
        )
        return {
            "messages": [AIMessage(content="I've reached my maximum reasoning iterations. Here's my best response based on the information gathered so far.")],
            "stop_reason": None  # Force END
        }
    
    # Build the prompt with tool context and memory context
    prompt = _messages_to_prompt(
        messages, 
        mandate=mandate, 
        tool_schemas=tool_schemas,
        memory_context=memory_context,  # Story 2.1
        user_model_context=user_model_context # Story 2.3
    )

    # --- Shadow Mode & Main Execution (Story 5.5) ---
    shadow_log_update = []
    reasoning_trace = ""
    stop_reason = None
    parsed_tool_calls = None

    async def execute_main_task():
        nonlocal reasoning_trace, stop_reason, parsed_tool_calls
        try:
            # Stream the LLM response
            async for chunk in stream_llm(prompt, purpose="reasoning"):
                reasoning_trace += chunk

                # Check for control tokens (legacy support)
                if "<fold_thought>" in reasoning_trace:
                    stop_reason = "fold_thought"
                    break
                if "<retrieve_tool>" in reasoning_trace:
                    stop_reason = "retrieve_tool"
                    break
            
            # If no control tokens, check for tool calls in the response
            if stop_reason is None:
                parsed_tool_calls = _parse_tool_calls_from_response(reasoning_trace)
                if parsed_tool_calls:
                    stop_reason = "tool_call"
                    core.logging.log_event(
                        f"Reasoning node detected {len(parsed_tool_calls)} tool call(s): {[tc['name'] for tc in parsed_tool_calls]}",
                        "INFO"
                    )
        except Exception as e:
            core.logging.log_event(f"Error in reasoning node: {e}", "ERROR")
            reasoning_trace = f"An error occurred during reasoning: {e}"

    async def execute_shadow_task():
        nonlocal shadow_log_update
        try:
            shadow_prompt = _apply_shadow_heuristics(prompt, messages)
            # Only run if the prompt was actually modified
            if shadow_prompt != prompt:
                core.logging.log_event("Running Shadow Mode reasoning (Parallel)...", "DEBUG")
                shadow_start = time.time()
                shadow_response = await run_llm(shadow_prompt, purpose="shadow_reasoning")
                shadow_duration = time.time() - shadow_start

                shadow_entry = {
                    "timestamp": shadow_start,
                    "duration": shadow_duration,
                    "original_prompt_len": len(prompt),
                    "shadow_prompt_len": len(shadow_prompt),
                    "result": shadow_response.get("result", "")
                }
                shadow_log_update = [shadow_entry] # Will be appended to list in state update
        except Exception as e:
            core.logging.log_event(f"Shadow Mode execution failed: {e}", "WARNING")

    tasks = [execute_main_task()]
    if shadow_mode:
        tasks.append(execute_shadow_task())
        
    await asyncio.gather(*tasks)
    
    # Construct the AIMessage with potential tool_calls
    if parsed_tool_calls:
        response_message = AIMessage(
            content=reasoning_trace,
            tool_calls=parsed_tool_calls
        )
    else:
        response_message = AIMessage(content=reasoning_trace)
    
    state_update = {
        "messages": [response_message],
        "stop_reason": stop_reason
    }

    if shadow_log_update:
        # Append to existing log
        current_log = state.get("shadow_log", [])
        state_update["shadow_log"] = current_log + shadow_log_update

    return state_update
