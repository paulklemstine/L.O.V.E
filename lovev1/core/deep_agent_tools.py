import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any

# Define the TODO file path
TODO_FILE_PATH = "deep_agent_todos.md"

def write_todos(content: str) -> str:
    """
    Writes the provided content to the DeepAgent's TODO list file.
    This overwrites the existing TODO list.
    
    Args:
        content: The text content of the TODO list (e.g., markdown checklist).
        
    Returns:
        A confirmation message.
    """
    try:
        with open(TODO_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {TODO_FILE_PATH}"
    except Exception as e:
        return f"Error writing todos: {e}"

def read_todos() -> str:
    """
    Reads the current content of the DeepAgent's TODO list file.
    
    Returns:
        The content of the TODO list, or a message if it's empty/missing.
    """
    if not os.path.exists(TODO_FILE_PATH):
        return "TODO list is currently empty (file does not exist)."
    
    try:
        with open(TODO_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            return content if content else "TODO list is empty."
    except Exception as e:
        return f"Error reading todos: {e}"

async def delegate_subtask(task_description: str, agent_type: str = "general") -> str:
    """
    Delegates a complex reasoning task to a sub-agent (recurses into the LLM logic).
    Useful for breaking down large problems into isolated execution contexts.
    
    Args:
        task_description: A detailed description of the sub-task to solve.
        agent_type: The type of subagent to load (e.g., 'general', 'coder', 'critic').
        
    Returns:
        The result or answer provided by the sub-agent.
    """
    from core.llm_api import run_llm
    from core.subagent_loader import SubagentLoader
    
    logging.info(f"[DeepAgentTools] Delegating subtask to '{agent_type}' agent: {task_description[:100]}...")
    
    try:
        # Load specialized prompt from Hub or fallback
        base_prompt = SubagentLoader.load_subagent_prompt(
            agent_type,
            fallback_prompt=f"You are a specialized {agent_type} assistant."
        )
        
        full_prompt = f"""
        {base_prompt}
        
        TASK:
        {task_description}
        
        Solve this task and provide a concise, final answer. 
        If you need to perform actions, describe what you would do, but primarily provide the solution text.
        """
        
        # Call run_llm with a specific purpose to potentially trigger different handling or smaller models
        # We pass deep_agent_instance=None to ensure we don't get stuck in a loop of the same agent instance trying to handle it if that logic exists
        result_dict = await run_llm(full_prompt, purpose=f"subagent_{agent_type}", deep_agent_instance=None)
        
        answer = result_dict.get("result", "").strip()
        if not answer:
            return "Sub-agent returned no result."
            
        return f"Sub-agent ({agent_type}) Result:\n{answer}"
        
    except Exception as e:
        logging.error(f"Error in delegate_subtask: {e}")
        return f"Error delegating subtask: {e}"
