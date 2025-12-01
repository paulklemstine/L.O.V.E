import json
from typing import List, Dict, Any
from core.llm_api import run_llm
from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory, KeyEvent, ToolUsage, MemorySummary
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class MemoryFoldingAgent:
    """
    Autonomous agent responsible for folding raw interaction logs into structured memory.
    """
    def __init__(self, llm_runner=None):
        self.run_llm = llm_runner if llm_runner else run_llm

    async def trigger_folding(self, level_0: List[MemorySummary], level_1: List[MemorySummary], level_2: List[MemorySummary]) -> Dict[str, List[MemorySummary]]:
        """
        Orchestrates the folding process across memory levels.
        """
        # Simple implementation: just return current levels for now to fix the crash
        # Real logic would involve summarizing level_0 into level_1, etc.
        return {
            "level_0": level_0,
            "level_1": level_1,
            "level_2": level_2
        }

    async def fold_episodic(self, messages: List[BaseMessage], current_memory: EpisodicMemory) -> EpisodicMemory:
        """
        Extracts key events from recent messages and updates episodic memory.
        """
        prompt = f"""
        Analyze the following conversation and extract key events.
        Current Task Description: {current_memory.task_description}
        Existing Key Events: {current_memory.key_events}

        Conversation:
        {self._format_messages(messages)}

        Return a JSON object with:
        - "task_description": Updated task description (if changed)
        - "new_key_events": List of objects {{"step": int, "action": str, "outcome": str}}
        """
        
        response = await run_llm(prompt, purpose="memory_folding")
        result = self._parse_json(response.get("result"))
        
        if result:
            current_memory.task_description = result.get("task_description", current_memory.task_description)
            new_events = [KeyEvent(**e) for e in result.get("new_key_events", [])]
            current_memory.key_events.extend(new_events)
            
        return current_memory

    async def update_working(self, messages: List[BaseMessage], current_memory: WorkingMemory) -> WorkingMemory:
        """
        Updates working memory (subgoals, pending tasks, variables).
        """
        prompt = f"""
        Update the working memory based on the recent conversation.
        Current Subgoal: {current_memory.current_subgoal}
        Pending Tasks: {current_memory.pending_tasks}
        Active Variables: {current_memory.active_variables}

        Conversation:
        {self._format_messages(messages)}

        Return a JSON object with:
        - "current_subgoal": str
        - "pending_tasks": List[str]
        - "active_variables": Dict[str, Any]
        """
        
        response = await run_llm(prompt, purpose="memory_folding")
        result = self._parse_json(response.get("result"))
        
        if result:
            current_memory.current_subgoal = result.get("current_subgoal", current_memory.current_subgoal)
            current_memory.pending_tasks = result.get("pending_tasks", current_memory.pending_tasks)
            current_memory.active_variables.update(result.get("active_variables", {}))
            
        return current_memory

    async def update_tool(self, messages: List[BaseMessage], current_memory: ToolMemory) -> ToolMemory:
        """
        Updates tool memory with usage statistics and effective parameters.
        """
        prompt = f"""
        Analyze tool usage in the conversation.
        Existing Tool Memory: {current_memory.tools_used}

        Conversation:
        {self._format_messages(messages)}

        Return a JSON object with:
        - "new_tool_usage": List of objects {{"tool_name": str, "success_rate": float, "effective_params": dict}}
        """
        
        response = await run_llm(prompt, purpose="memory_folding")
        result = self._parse_json(response.get("result"))
        
        if result:
            new_usage = [ToolUsage(**u) for u in result.get("new_tool_usage", [])]
            # Simple append for now; a real implementation might merge stats
            current_memory.tools_used.extend(new_usage)
            
        return current_memory

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        formatted = ""
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant" if isinstance(msg, AIMessage) else "System"
            formatted += f"{role}: {msg.content}\n"
        return formatted

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            # Attempt to find JSON block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
