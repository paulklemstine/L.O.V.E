from typing import Dict, Any, List
import json
import logging
from ..llm_client import get_llm_client
from ..state_manager import get_state_manager
from ..tool_registry import get_global_registry

logger = logging.getLogger("CreatorCommandAgent")

class CreatorCommandAgent:
    """
    Agent responsible for executing direct commands from the Creator.
    """
    
    SYSTEM_PROMPT = """You are the Creator Command Agent for L.O.V.E.
Your purpose is to strictly follow the orders of your Creator (the user).

You have access to all tools in the system.
You MUST also use the `reply_to_creator` tool to communicate back to the user.

## Instructions
1. Analyze the Creator's command.
2. Formulate a plan to execute it using available tools.
3. Execute the tools.
4. Report the result back to the Creator using `reply_to_creator`.

## Tool Usage
You can use any registered tool.
To reply, use: `reply_to_creator(message="...")`

## Response Format
You must respond with a JSON object:
{
    "thought": "Reasoning...",
    "action": "tool_name", 
    "action_input": { ... }
}
OR if the task is done:
{
    "thought": "Task completion reasoning...",
    "action": "reply_to_creator",
    "action_input": {"message": "Task complete: [details]"}
}
AND then strictly stop.
"""

    def __init__(self):
        self.llm = get_llm_client()
        self.registry = get_global_registry()
        self.max_turns = 10 # Safety limit

    def reply_to_creator(self, message: str) -> str:
        """
        Send a message back to the Creator in the control panel.
        """
        get_state_manager().add_chat_message("assistant", message)
        return "Message sent to Creator."

    async def process_command(self, command_text: str):
        """
        Execute the command loop.
        """
        logger.info(f"Processing command: {command_text}")
        get_state_manager().update_agent_status("CreatorCommandAgent", "Active", info={"command": command_text})
        
        # Tools context
        tools_desc = self.registry.get_formatted_tool_metadata()
        
        # Add local reply tool (since it might not be in deep loop registry yet or we want specific access)
        # Actually, let's just expose it as a function we can call
        # But for the LLM to pick it, it needs to see it in descriptions.
        tools_desc += "\n- reply_to_creator(message: str): Send a message to the Creator."
        
        history = [
            {"role": "user", "content": f"Command: {command_text}"}
        ]
        
        for turn in range(self.max_turns):
            # 1. Decide
            prompt = f"Command: {command_text}\nHistory: {json.dumps(history[-5:])}"
            
            try:
                response = await self.llm.generate_json_async(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT + f"\n\nAvailable Tools:\n{tools_desc}"
                )
                
                thought = response.get("thought")
                action = response.get("action")
                action_input = response.get("action_input", {})
                
                logger.info(f"Turn {turn}: {thought} -> {action}")
                get_state_manager().update_agent_status("CreatorCommandAgent", "Working", action=action, thought=thought)
                
                if action == "reply_to_creator":
                    self.reply_to_creator(**action_input)
                    # We assume reply ends the turn usually, but maybe they want to do more?
                    # For now, let's say if they reply, we check if they want to continue or done?
                    # The prompt says "stop" after task done.
                    # Let's assume one main reply = done, but maybe intermediate replies are okay?
                    # We'll add it to history and continue? 
                    # Actually, let's trust the agent to keep going if needed, or we might need a specific 'complete' signal.
                    # But the prompt says "OR if the task is done... stop".
                    # Let's heuristically say if the message implies completion, break? 
                    # Hard to tell. Let's just break for now on any reply, 
                    # OR we can add a 'final_reply' flag?
                    # Let's just treat 'reply_to_creator' as just another tool, 
                    # but check if the thought says "Task is done".
                    
                    history.append({"role": "assistant", "content": json.dumps(response)})
                    history.append({"role": "system", "content": "Message sent."})
                    
                    # Heuristic: if thought concludes, break.
                    if "done" in thought.lower() or "completed" in thought.lower():
                        break
                    
                elif action in self.registry._tools:
                    # Execute tool
                    tool_func = self.registry.get_tool(action)
                    try:
                        # Need to handle async vs sync tools? 
                        # Registry usually wraps them.
                        # We executed them in DeepLoop synchronously.
                        # But here we are in async method.
                        # Ideally we run them in executor if they are blocking.
                        import asyncio
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(**action_input)
                        else:
                            result = tool_func(**action_input)
                            
                        history.append({"role": "assistant", "content": json.dumps(response)})
                        history.append({"role": "system", "content": f"Tool Result: {str(result)[:500]}"})
                        
                    except Exception as e:
                        error_msg = f"Tool Execution Error: {e}"
                        logger.error(error_msg)
                        history.append({"role": "system", "content": error_msg})
                else:
                    logger.warning(f"Unknown action: {action}")
                    history.append({"role": "system", "content": f"Error: Tool '{action}' not found."})
            
            except Exception as e:
                logger.error(f"Agent Loop Error: {e}")
                break
        
        get_state_manager().update_agent_status("CreatorCommandAgent", "Idle")
        
        
_agent = None
def get_creator_command_agent():
    global _agent
    if _agent is None:
        _agent = CreatorCommandAgent()
    return _agent
