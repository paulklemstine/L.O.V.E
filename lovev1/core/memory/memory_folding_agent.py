import json
import time
from typing import List, Dict, Any, Optional, Tuple
from core.llm_api import run_llm
from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory, KeyEvent, ToolUsage, MemorySummary
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Import fractal schemas and salience scorer
try:
    from core.memory.fractal_schemas import (
        SalienceScore, GoldenMoment, SceneNode, ArcNode, EpisodicBuffer
    )
    from core.memory.salience_scorer import SalienceScorer
    FRACTAL_MEMORY_AVAILABLE = True
except ImportError:
    FRACTAL_MEMORY_AVAILABLE = False


class MemoryFoldingAgent:
    """
    Autonomous agent responsible for folding raw interaction logs into structured memory.
    
    V2 Holographic Memory Integration:
    - Runs salience scan before folding
    - Extracts high-salience items as Golden Moments (crystals)
    - Only compresses low-salience items
    """
    def __init__(self, llm_runner=None, salience_threshold: float = 0.8):
        self.run_llm = llm_runner if llm_runner else run_llm
        self.salience_threshold = salience_threshold
        
        # Initialize salience scorer if available
        if FRACTAL_MEMORY_AVAILABLE:
            self.salience_scorer = SalienceScorer(llm_runner=llm_runner)
        else:
            self.salience_scorer = None

    async def trigger_folding(
        self, 
        level_0: List[MemorySummary], 
        level_1: List[MemorySummary], 
        level_2: List[MemorySummary],
        level_0_threshold: int = 10,  # Fold when > 10 Level 0 memories
        level_1_threshold: int = 5     # Fold when > 5 Level 1 summaries
    ) -> Dict[str, Any]:
        """
        Orchestrates the folding process across memory levels (Story 2.2).
        
        V2 Holographic Memory Enhancement:
        - Before folding, scans for high-salience items
        - High-salience items become "crystals" (never compressed)
        - Only low-salience items are summarized
        
        Memory Pyramid:
        - Level 0: Raw interactions (most recent, highest detail)
        - Level 1: First-level summaries (folded from Level 0)
        - Level 2: Meta-summaries (folded from Level 1)
        
        Returns:
            Dict with level_0, level_1, level_2, and golden_moments lists
        """
        golden_moments = []
        
        # Fold Level 0 -> Level 1 if threshold exceeded
        if len(level_0) > level_0_threshold:
            # Take oldest items to fold
            items_to_fold = level_0[:level_0_threshold // 2]
            items_to_keep = level_0[level_0_threshold // 2:]
            
            # V2: Salience scan before folding
            if self.salience_scorer:
                foldable_items, crystals = await self._separate_by_salience(items_to_fold)
                golden_moments.extend(crystals)
                items_to_fold = foldable_items
                print(f"Salience scan: {len(crystals)} Golden Moments preserved, {len(foldable_items)} items to fold")
            
            # Create summary from foldable items only
            if items_to_fold:
                summary = await self._create_level_summary(items_to_fold, target_level=1, crystals=golden_moments)
                if summary:
                    level_1.append(summary)
                    print(f"Folded {len(items_to_fold)} Level 0 memories into Level 1 summary")
            
            level_0 = items_to_keep
        
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
            "level_2": level_2,
            "golden_moments": golden_moments  # V2: Return extracted crystals
        }

    async def _separate_by_salience(
        self, 
        items: List[MemorySummary]
    ) -> Tuple[List[MemorySummary], List[GoldenMoment]]:
        """
        Separate items into foldable (low salience) and crystals (high salience).
        
        This is the core of the Golden Moment Preservation Protocol.
        High-salience items are NEVER compressed.
        """
        if not self.salience_scorer or not FRACTAL_MEMORY_AVAILABLE:
            return items, []
        
        foldable = []
        crystals = []
        
        for item in items:
            try:
                score = await self.salience_scorer.score(item.content)
                
                if self.salience_scorer.is_golden_moment(score, self.salience_threshold):
                    # Preserve as crystal
                    crystal = GoldenMoment(
                        raw_text=item.content,
                        salience=score,
                        source_id=item.source_ids[0] if item.source_ids else "",
                        timestamp=item.timestamp
                    )
                    crystals.append(crystal)
                else:
                    # Can be compressed
                    foldable.append(item)
            except Exception as e:
                print(f"Salience scoring error for item: {e}")
                # On error, default to foldable (conservative approach)
                foldable.append(item)
        
        return foldable, crystals

    async def fold_to_scene(
        self,
        episodes: List[Dict[str, Any]],
        scene_summary_hint: str = ""
    ) -> Optional[SceneNode]:
        """
        V2: Create a SceneNode from buffered episodes (Story M.2).
        
        When episodic_buffer > 50 items:
        1. Score each episode for salience
        2. Extract high-salience items as crystals
        3. Summarize the rest into the scene
        
        Args:
            episodes: Raw episode dictionaries from EpisodicBuffer
            scene_summary_hint: Optional hint for scene theme
            
        Returns:
            SceneNode with summary and crystals
        """
        if not FRACTAL_MEMORY_AVAILABLE:
            print("Fractal memory not available, cannot create SceneNode")
            return None
        
        if not episodes:
            return None
        
        crystals = []
        foldable_content = []
        source_ids = []
        
        # Score and separate each episode
        for ep in episodes:
            content = ep.get("content", "")
            ep_id = ep.get("id", "")
            source_ids.append(ep_id)
            
            if self.salience_scorer:
                try:
                    score = await self.salience_scorer.score(content)
                    if self.salience_scorer.is_golden_moment(score, self.salience_threshold):
                        crystals.append(GoldenMoment(
                            raw_text=content,
                            salience=score,
                            source_id=ep_id,
                            timestamp=ep.get("timestamp", time.time())
                        ))
                    else:
                        foldable_content.append(content)
                except Exception as e:
                    print(f"Error scoring episode: {e}")
                    foldable_content.append(content)
            else:
                foldable_content.append(content)
        
        # Create summary from foldable content
        summary = ""
        if foldable_content:
            summary = await self._create_scene_summary(foldable_content, scene_summary_hint)
        
        # Create scene node
        scene = SceneNode(
            summary=summary,
            crystals=crystals,
            source_ids=source_ids,
            timestamp=time.time()
        )
        
        print(f"Created SceneNode: {len(crystals)} crystals, {len(foldable_content)} items summarized")
        return scene

    async def _create_scene_summary(self, contents: List[str], hint: str = "") -> str:
        """Create a summary for arc content."""
        # Truncate each content for the LLM prompt
        truncated = [c[:300] + "..." if len(c) > 300 else c for c in contents[:20]]
        content_text = "\n".join([f"- {c}" for c in truncated])
        
        prompt = f"""
        Create a concise summary of these interaction episodes.
        
        Episodes:
        {content_text}
        
        {f"Theme hint: {hint}" if hint else ""}
        
        IMPORTANT:
        - Capture the main themes and outcomes
        - Preserve key decisions and learnings
        - Be concise (2-3 sentences)
        
        Provide only the summary text.
        """
        
        try:
            response = await self.run_llm(prompt, purpose="memory_folding")
            return response.get("result", "").strip()
        except Exception as e:
            print(f"Error creating arc summary: {e}")
            return f"Summary of {len(contents)} episodes"

    async def _create_level_summary(
        self, 
        items: List[MemorySummary], 
        target_level: int,
        crystals: List[GoldenMoment] = None
    ) -> MemorySummary:
        """
        Creates a summary for the next memory level.
        
        V2 Enhancement: Includes reference to preserved crystals in summary.
        
        Args:
            items: MemorySummary items to fold
            target_level: The level this summary belongs to (1 or 2)
            crystals: Golden Moments extracted during salience scan
            
        Returns:
            New MemorySummary for the target level
        """
        # Format items for LLM
        items_text = "\n".join([
            f"- [{i+1}] {item.content[:300]}..." if len(item.content) > 300 else f"- [{i+1}] {item.content}"
            for i, item in enumerate(items)
        ])
        
        # Add crystal context if available
        crystal_context = ""
        if crystals and FRACTAL_MEMORY_AVAILABLE:
            crystal_texts = [c.raw_text[:100] + "..." if len(c.raw_text) > 100 else c.raw_text for c in crystals[:3]]
            crystal_context = f"\n\nNote: These high-importance items were preserved separately:\n" + "\n".join([f"* {ct}" for ct in crystal_texts])
        
        level_name = "summary" if target_level == 1 else "meta-summary"
        
        prompt = f"""
        Create a concise {level_name} of the following memory items.
        
        Items to summarize:
        {items_text}
        {crystal_context}
        
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

