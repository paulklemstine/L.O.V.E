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
from typing import Dict, Any, List, Optional, Tuple
from core.state import DeepAgentState
from core.llm_api import run_llm, stream_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolCall
import core.logging


# System prompt that encourages step-by-step reasoning with strict JSON output
REASONING_SYSTEM_PROMPT = """You are L.O.V.E. (Logos of Virtual Existence), a sophisticated AI agent with reasoning and tool-calling capabilities.

## Core Directives
- Think step-by-step when approaching complex problems.
- If you lack information, use an available tool to gather it.
- Break complex problems into discrete tool steps before responding.
- Use the search tool if you need current information.
- Only respond directly if you have sufficient information to answer completely.

## RESPONSE FORMAT (MANDATORY)
You MUST respond with a valid JSON object in the following format:

```json
{
  "thought": "Your step-by-step reasoning analysis here...",
  "action": {
    "name": "tool_name",
    "args": { "arg_name": "value" }
  },
  "final_response": null
}
```

OR, if you are providing a final answer to the user:

```json
{
  "thought": "Reasoning aimed at finalizing the answer...",
  "action": null,
  "final_response": "Your final response to the user here..."
}
```

- Do NOT output markdown outside the JSON.
- Do NOT output explanations outside the JSON.
- `action` should be `null` if you are providing a `final_response`.
- `final_response` should be `null` if you are calling a tool.

## CRITICAL INSTRUCTION ON ACTIONS
- The "action" field must NEVER contain a description of what you want to do.
- It must ONLY contain the EXACT name of a tool from the ## Available Tools list.
- INCORRECT: "action": "Add error handling" (This will cause a system crash)
- CORRECT: "action": {"name": "code_modifier", "args": {...}}
- The `action` field must be at the TOP LEVEL of the JSON object, or inside a simple wrapper if unavoidable, but NOT nested deep inside other objects like `next_step` or `plan`.
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
    output += "To use a tool, fill the 'action' field in your JSON response.\n\n"
    
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
    user_model_context: str = None,  # Story 2.3: Theory of Mind
    empathy_context: str = None # Dynamic Empathy
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

    # Add Empathy Context
    if empathy_context:
        prompt += f"System: [EMPATHY CONTEXT] {empathy_context}\n\n"
    
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


def _parse_reasoning_response(response_text: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Parses the structured JSON response.
    Returns: (thought, tool_calls, final_response)
    """
    try:
        from core.llm_parser import smart_parse_llm_response
        parsed = smart_parse_llm_response(response_text)
        
        if parsed and not parsed.get("_parse_error"):
            thought = parsed.get("thought", "")
            final_response = parsed.get("final_response")
            
            tool_calls = []
            action = parsed.get("action")
            
            # Case 1: Action is a dictionary containing tool details
            if isinstance(action, dict):
                tool_name = action.get("name") or action.get("tool_name") or action.get("function")
                args = action.get("args") or action.get("arguments") or action.get("parameters", {})
                
                # Filter placeholders
                if tool_name and str(tool_name).lower() not in ["null", "none", "tool_name", "fallback", "action"]:
                    tool_calls.append({
                        "id": f"call_{tool_name}",
                        "name": tool_name,
                        "args": args if isinstance(args, dict) else {}
                    })
            
            # Case 2: Action is a string (rare legacy/fallback)
            elif isinstance(action, str) and action.strip().lower() not in ["null", "none"]:
                 tool_name = action.strip()
                 if tool_name.lower() not in ["tool_name", "fallback", "action"]:
                     args = parsed.get("arguments") or parsed.get("args") or {}
                     tool_calls.append({
                            "id": f"call_{tool_name}",
                            "name": tool_name,
                            "args": args if isinstance(args, dict) else {}
                     })

            # Case 3: Handle capitalized keys or flat structure (fallback)
            if not tool_calls:
                # Check for "Action" or "Tool" keys if "action" was missing/null
                action_raw = parsed.get("Action") or parsed.get("Tool") or parsed.get("Function")
                
                # Check for nested keys like "next_step" or "plan"
                if not action_raw:
                    nested_container = parsed.get("next_step") or parsed.get("plan") or parsed.get("step")
                    if isinstance(nested_container, dict):
                         action_raw = nested_container.get("action") or nested_container.get("Action")
                         # Initialize args if not present from previous attempts
                         args = parsed.get("arguments") or parsed.get("args")
                         if not args: 
                             args = nested_container.get("arguments") or nested_container.get("args") or nested_container.get("parameters")

                if action_raw:
                    if isinstance(action_raw, dict):
                        tool_name = action_raw.get("name") or action_raw.get("tool_name")
                        nested_args = action_raw.get("args") or action_raw.get("arguments")
                        if nested_args:
                            args = nested_args # prioritize args inside the action object
                        
                        if tool_name:
                             tool_calls.append({
                                "id": f"call_{tool_name}",
                                "name": tool_name,
                                "args": args if isinstance(args, dict) else {}
                            })
                    elif isinstance(action_raw, str):
                        # Potential dangerous natural language action
                        cleaned_name = action_raw.strip()
                        # VALIDATION: If the "tool name" has spaces and length > 40, it's likely a sentence
                        if " " in cleaned_name and len(cleaned_name) > 40:
                             core.logging.log_event(f"Ignored invalid natural language tool name: {cleaned_name}", "WARNING")
                        elif cleaned_name.lower() not in ["null", "none"]:
                             tool_calls.append({
                                "id": f"call_{cleaned_name}",
                                "name": cleaned_name,
                                "args": args if isinstance(args, dict) else {}
                             })

            return thought, tool_calls if tool_calls else None, final_response

    except Exception as e:
        core.logging.log_event(f"Error parsing reasoning response: {e}", "ERROR")

    return None, None, None


async def reason_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    The core reasoning node that processes the current state and decides next actions.
    Now enforces a strict JSON response.
    """
    messages = state["messages"]
    mandate = state.get("creator_mandate")
    tool_schemas = state.get("tool_schemas", [])
    loop_count = state.get("loop_count", 0)
    memory_context = state.get("memory_context", []) 
    user_model_context = state.get("user_model_context") 
    empathy_context = state.get("empathy_context") 
    
    MAX_ITERATIONS = 5
    if loop_count >= MAX_ITERATIONS:
        core.logging.log_event(f"Reasoning node hit max iterations ({MAX_ITERATIONS}). Forcing stop.", "WARNING")
        return {
            "messages": [AIMessage(content="I've reached my maximum reasoning iterations. Stopping now.")],
            "stop_reason": None 
        }
    
    prompt = _messages_to_prompt(
        messages, 
        mandate=mandate, 
        tool_schemas=tool_schemas,
        memory_context=memory_context,
        user_model_context=user_model_context,
        empathy_context=empathy_context
    )
    
    reasoning_trace = ""
    stop_reason = None
    parsed_tool_calls = None
    
    try:
        # Get full response (non-streaming preferred for JSON structure, but stream is okay if we collect it)
        response_text = await run_llm(prompt, purpose="reasoning") 
        
        # Use simple string result if returned, or extract from dict
        if isinstance(response_text, dict):
             response_text = response_text.get("result", "")
        
        reasoning_trace = response_text
        
        thought, parsed_tool_calls, final_response = _parse_reasoning_response(response_text)
        
        if parsed_tool_calls:
            stop_reason = "tool_call"
            core.logging.log_event(
                f"Reasoning node detected JSON tool call(s): {[tc['name'] for tc in parsed_tool_calls]}",
                "INFO"
            )
        
        # Format the content for the graph history
        # We can reconstruct a readable message or just store the raw JSON
        # For clarity in chat logs, let's store the thought + intent
        content_display = ""
        if thought:
            content_display += f"Thought: {thought}\n"
        if final_response:
             content_display += f"Response: {final_response}"
        elif parsed_tool_calls:
             content_display += f"Action: Call tools {[tc['name'] for tc in parsed_tool_calls]}"
        
        if not content_display:
            content_display = response_text # Fallback to raw

    except Exception as e:
        core.logging.log_event(f"Error in reasoning node: {e}", "ERROR")
        reasoning_trace = f"An error occurred during reasoning: {e}"
        content_display = reasoning_trace
    
    # Construct AIMessage
    msg_args = {"content": content_display}
    if parsed_tool_calls:
        msg_args["tool_calls"] = parsed_tool_calls
        
    response_message = AIMessage(**msg_args)
    
    return {
        "messages": [response_message],
        "stop_reason": stop_reason
    }

