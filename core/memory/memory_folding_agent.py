# core/memory/memory_folding_agent.py

import json
import re
from typing import List, Dict
from dataclasses import dataclass, field
import uuid

@dataclass
class MemorySummary:
    """Represents a folded memory summary at a specific level."""
    content: str
    level: int  # 0 (raw), 1 (folded), 2 (meta)
    source_ids: List[str]  # IDs of memories/summaries that were folded into this
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: __import__('time').time())

class MemoryFoldingAgent:
    """
    Implements hierarchical memory compression using Chain of Density (CoD) summarization.
    Creates a 3-level pyramid: Level 0 (raw) → Level 1 (folded) → Level 2 (meta).
    """
    
    def __init__(self, llm_runner):
        """
        Args:
            llm_runner: Async function to call LLM, signature: async def run_llm(prompt) -> dict
        """
        self.llm_runner = llm_runner
        
        # Folding thresholds
        self.level_0_threshold = 10  # Fold when 10+ Level 0 items
        self.level_1_threshold = 10  # Fold when 10+ Level 1 items
        self.level_2_max = 20  # Keep only 20 most recent Level 2 summaries
        
    async def compress_chain_with_cod(self, items: List[str]) -> str:
        """
        Compresses a chain of interactions using Chain of Density (CoD) technique.
        
        CoD Process:
        1. Initial summary (under 50 words)
        2. Identify 3 missing key entities
        3. Rewrite to include entities (under 60 words)
        
        Args:
            items: List of interaction strings to compress
            
        Returns:
            Dense summary string
        """
        # Combine items into a single text
        combined_text = "\n---\n".join(items)
        
        # Step 1: Initial summary
        step1_prompt = f"""Summarize the following interaction chain in under 50 words, focusing on key actions and outcomes.

Interaction Chain:
{combined_text}

Provide ONLY the summary, no other text."""

        try:
            response_dict = await self.llm_runner(step1_prompt)
            initial_summary = response_dict.get("result", "").strip()
            
            # Step 2: Identify missing entities
            step2_prompt = f"""Given this summary:
"{initial_summary}"

And the original interactions:
{combined_text}

Identify 3 key entities (filenames, specific errors, tool names, function names) that are missing from the summary but are critical for understanding the interaction.

Respond with ONLY a JSON array of 3 entity strings, for example: ["config.json", "PermissionError", "write_file"]"""

            response_dict = await self.llm_runner(step2_prompt)
            entities_str = response_dict.get("result", "[]").strip()
            
            # Extract JSON from response
            match = re.search(r'\[.*?\]', entities_str, re.DOTALL)
            if match:
                entities_str = match.group(0)
            
            entities = json.loads(entities_str)
            
            # Step 3: Dense rewrite
            step3_prompt = f"""Rewrite this summary to include these entities while keeping the total length under 60 words. Prioritize specificity and actionable details.

Original Summary:
"{initial_summary}"

Entities to Include:
{', '.join(entities)}

Provide ONLY the rewritten summary, no other text."""

            response_dict = await self.llm_runner(step3_prompt)
            dense_summary = response_dict.get("result", "").strip()
            
            return dense_summary
            
        except Exception as e:
            print(f"Error in CoD compression: {e}")
            # Fallback: return a simple truncated version
            return combined_text[:200] + "..." if len(combined_text) > 200 else combined_text
    
    async def fold_level_0_to_level_1(self, level_0_items: List[MemorySummary]) -> MemorySummary:
        """
        Folds 5-10 Level 0 (raw) items into a single Level 1 (folded) summary.
        
        Args:
            level_0_items: List of Level 0 MemorySummary objects
            
        Returns:
            Level 1 MemorySummary
        """
        # Extract content from items
        contents = [item.content for item in level_0_items]
        
        # Compress using CoD
        dense_summary = await self.compress_chain_with_cod(contents)
        
        # Create Level 1 summary
        return MemorySummary(
            content=dense_summary,
            level=1,
            source_ids=[item.id for item in level_0_items]
        )
    
    async def fold_level_1_to_level_2(self, level_1_items: List[MemorySummary]) -> MemorySummary:
        """
        Folds 5-10 Level 1 (folded) summaries into a single Level 2 (meta) summary.
        
        Args:
            level_1_items: List of Level 1 MemorySummary objects
            
        Returns:
            Level 2 MemorySummary
        """
        # Extract content from items
        contents = [item.content for item in level_1_items]
        
        # Compress using CoD
        dense_summary = await self.compress_chain_with_cod(contents)
        
        # Create Level 2 summary
        return MemorySummary(
            content=dense_summary,
            level=2,
            source_ids=[item.id for item in level_1_items]
        )
    
    async def trigger_folding(self, 
                             level_0_memories: List[MemorySummary],
                             level_1_summaries: List[MemorySummary],
                             level_2_summaries: List[MemorySummary]) -> Dict[str, List[MemorySummary]]:
        """
        Automatically triggers folding when thresholds are met.
        
        Returns:
            Dictionary with updated memory levels: {
                'level_0': [...],
                'level_1': [...],
                'level_2': [...]
            }
        """
        # Check Level 0 → Level 1 folding
        if len(level_0_memories) >= self.level_0_threshold:
            # Take oldest 10 items for folding
            items_to_fold = level_0_memories[:10]
            remaining_level_0 = level_0_memories[10:]
            
            # Create Level 1 summary
            level_1_summary = await self.fold_level_0_to_level_1(items_to_fold)
            level_1_summaries.append(level_1_summary)
            
            print(f"Folded {len(items_to_fold)} Level 0 items into Level 1 summary {level_1_summary.id}")
            
            level_0_memories = remaining_level_0
        
        # Check Level 1 → Level 2 folding
        if len(level_1_summaries) >= self.level_1_threshold:
            # Take oldest 10 items for folding
            items_to_fold = level_1_summaries[:10]
            remaining_level_1 = level_1_summaries[10:]
            
            # Create Level 2 summary
            level_2_summary = await self.fold_level_1_to_level_2(items_to_fold)
            level_2_summaries.append(level_2_summary)
            
            print(f"Folded {len(items_to_fold)} Level 1 summaries into Level 2 summary {level_2_summary.id}")
            
            level_1_summaries = remaining_level_1
        
        # Cleanup Level 2: keep only most recent items
        if len(level_2_summaries) > self.level_2_max:
            level_2_summaries = level_2_summaries[-self.level_2_max:]
            print(f"Cleaned up Level 2 summaries, keeping {self.level_2_max} most recent")
        
        return {
            'level_0': level_0_memories,
            'level_1': level_1_summaries,
            'level_2': level_2_summaries
        }
