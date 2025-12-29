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

    async def trigger_folding(
        self, 
        level_0: List[MemorySummary], 
        level_1: List[MemorySummary], 
        level_2: List[MemorySummary],
        level_0_threshold: int = 10,  # Fold when > 10 Level 0 memories
        level_1_threshold: int = 5     # Fold when > 5 Level 1 summaries
    ) -> Dict[str, List[MemorySummary]]:
        """
        Orchestrates the folding process across memory levels (Story 2.2).
        
        Memory Pyramid:
        - Level 0: Raw interactions (most recent, highest detail)
        - Level 1: First-level summaries (folded from Level 0)
        - Level 2: Meta-summaries (folded from Level 1)
        
        Triggers:
        - Level 0 -> Level 1: When len(level_0) > level_0_threshold
        - Level 1 -> Level 2: When len(level_1) > level_1_threshold
        """
        import time
        
        # Fold Level 0 -> Level 1 if threshold exceeded
        if len(level_0) > level_0_threshold:
            # Take oldest items to fold
            items_to_fold = level_0[:level_0_threshold // 2]
            items_to_keep = level_0[level_0_threshold // 2:]
            
            # Create summary
            summary = await self._create_level_summary(items_to_fold, target_level=1)
            if summary:
                level_1.append(summary)
                level_0 = items_to_keep
                print(f"Folded {len(items_to_fold)} Level 0 memories into Level 1 summary")
        
        # Fold Level 1 -> Level 2 if threshold exceeded
        if len(level_1) > level_1_threshold:
            # Take oldest items to fold
            items_to_fold = level_1[:level_1_threshold // 2]
            items_to_keep = level_1[level_1_threshold // 2:]
            
            # Create meta-summary
            summary = await self._create_level_summary(items_to_fold, target_level=2)
            if summary:
                level_2.append(summary)
                level_1 = items_to_keep
                print(f"Folded {len(items_to_fold)} Level 1 summaries into Level 2 meta-summary")
        
        return {
            "level_0": level_0,
            "level_1": level_1,
            "level_2": level_2
        }

    async def _create_level_summary(
        self, 
        items: List[MemorySummary], 
        target_level: int
    ) -> MemorySummary:
        """
        Creates a summary for the next memory level.
        
        Args:
            items: MemorySummary items to fold
            target_level: The level this summary belongs to (1 or 2)
            
        Returns:
            New MemorySummary for the target level
        """
        import time
        
        # Format items for LLM
        items_text = "\n".join([
            f"- [{i+1}] {item.content[:300]}..." if len(item.content) > 300 else f"- [{i+1}] {item.content}"
            for i, item in enumerate(items)
        ])
        
        level_name = "summary" if target_level == 1 else "meta-summary"
        
        prompt = f"""
        Create a concise {level_name} of the following memory items.
        
        Items to summarize:
        {items_text}
        
        IMPORTANT:
        - Preserve key insights and learnings
        - Maintain any critical directives or goals
        - Be concise but comprehensive
        - Focus on patterns and outcomes
        
        Provide only the {level_name} text, no preamble.
        """
        
        try:
            response = await self.run_llm(prompt, purpose="memory_folding")
            summary_text = response.get("result", "").strip()
            
            if not summary_text:
                return None
            
            return MemorySummary(
                content=summary_text,
                level=target_level,
                source_ids=[item.source_ids[0] if item.source_ids else "" for item in items],
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Error creating level {target_level} summary: {e}")
            return None

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
