"""
autonomous_memory_folding.py - Context Compression via Memory Folding

When context grows too large, this module "folds" the interaction history
by summarizing it into a condensed form, allowing fresh reasoning while
preserving critical information.

Inspired by DeepAgent's Autonomous Memory Folding mechanism.

See docs/autonomous_memory_folding.md for detailed documentation.
"""

import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .llm_client import get_llm_client


@dataclass
class FoldedMemory:
    """Represents a folded (compressed) memory segment."""
    timestamp: str
    original_length: int
    summary: str
    key_decisions: List[str]
    key_learnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "original_length": self.original_length,
            "summary": self.summary,
            "key_decisions": self.key_decisions,
            "key_learnings": self.key_learnings
        }


class AutonomousMemoryFolder:
    """
    Compresses long interaction histories into structured summaries.
    
    When context length approaches limits, the folder:
    1. Identifies the middle portion of context (preserving recent + earliest)
    2. Summarizes that portion using an LLM
    3. Returns a compressed version that maintains key information
    """
    
    # Thresholds for triggering folding
    DEFAULT_MAX_TOKENS = 4096  # Safer default for local models
    FOLD_TRIGGER_RATIO = 0.75  # Fold when context is 75% of max
    
    # Prompts for the LLM summarizer
    FOLD_SYSTEM_PROMPT = """You are a memory compression agent. Your task is to summarize 
interaction history while preserving critical information.

Focus on:
1. Key decisions made
2. Important results/outcomes
3. Lessons learned
4. Active goals and their progress

Be concise but comprehensive. Output valid JSON."""
    
    FOLD_USER_PROMPT_TEMPLATE = """Compress the following interaction history into a structured summary.

INTERACTION HISTORY:
{history}

Respond with JSON in this exact format:
{{
    "summary": "Brief overall summary (2-3 sentences)",
    "key_decisions": ["decision 1", "decision 2", ...],
    "key_learnings": ["learning 1", "learning 2", ...],
    "active_context": "Any context needed for continuation"
}}"""
    
    def __init__(
        self, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fold_ratio: float = FOLD_TRIGGER_RATIO
    ):
        """
        Initialize the memory folder.
        
        Args:
            max_tokens: Maximum context tokens before folding.
            fold_ratio: Ratio of max_tokens that triggers folding.
        """
        self.max_tokens = max_tokens
        self.fold_ratio = fold_ratio
        self.fold_history: List[FoldedMemory] = []
        self.llm = get_llm_client()
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Rough approximation: ~4 characters per token for English.
        """
        return len(text) // 4
    
    def should_fold(self, context: str) -> bool:
        """Check if context needs folding."""
        estimated_tokens = self.estimate_tokens(context)
        
        # Check if LLM client has detected a specific model length
        current_max = self.max_tokens
        if hasattr(self.llm, 'max_model_len') and self.llm.max_model_len:
             current_max = self.llm.max_model_len

        threshold = int(current_max * self.fold_ratio)
        return estimated_tokens > threshold
    
    def fold(self, context: str, preserve_recent: int = 1000) -> str:
        """
        Fold (compress) the context if needed.
        
        Args:
            context: The full context string to potentially fold.
            preserve_recent: Number of characters to preserve from the end.
            
        Returns:
            Original context if no folding needed, otherwise compressed version.
        """
        if not self.should_fold(context):
            return context
        
        # Split context: preserve beginning (system context) and end (recent)
        preserved_start = context[:500]  # Keep first 500 chars (usually system info)
        preserved_end = context[-preserve_recent:]
        
        # Middle portion to be summarized
        middle = context[500:-preserve_recent] if len(context) > 500 + preserve_recent else ""
        
        if not middle:
            return context
        
        # Summarize the middle portion
        try:
            summary_json = self.llm.generate_json(
                prompt=self.FOLD_USER_PROMPT_TEMPLATE.format(history=middle),
                system_prompt=self.FOLD_SYSTEM_PROMPT,
                temperature=0.3
            )
            
            # Create folded memory record
            folded = FoldedMemory(
                timestamp=datetime.now().isoformat(),
                original_length=len(middle),
                summary=summary_json.get("summary", ""),
                key_decisions=summary_json.get("key_decisions", []),
                key_learnings=summary_json.get("key_learnings", [])
            )
            self.fold_history.append(folded)
            
            # Construct compressed context
            compressed_middle = self._format_folded_memory(folded)
            
            return f"{preserved_start}\n\n{compressed_middle}\n\n{preserved_end}"
            
        except Exception as e:
            print(f"[MemoryFolder] Folding failed: {e}")
            # On failure, just truncate the middle
            return f"{preserved_start}\n\n[...context compressed...]\n\n{preserved_end}"
    
    def _format_folded_memory(self, folded: FoldedMemory) -> str:
        """Format a folded memory for inclusion in context."""
        lines = [
            "=== COMPRESSED HISTORY ===",
            f"(Original: {folded.original_length} chars, folded at {folded.timestamp})",
            "",
            f"**Summary**: {folded.summary}",
        ]
        
        if folded.key_decisions:
            lines.append("\n**Key Decisions**:")
            for decision in folded.key_decisions[:5]:
                lines.append(f"  • {decision}")
        
        if folded.key_learnings:
            lines.append("\n**Key Learnings**:")
            for learning in folded.key_learnings[:5]:
                lines.append(f"  • {learning}")
        
        lines.append("=== END COMPRESSED ===")
        
        return "\n".join(lines)
    
    def get_fold_count(self) -> int:
        """Get the number of times folding has occurred."""
        return len(self.fold_history)
    
    def get_total_compressed(self) -> int:
        """Get total characters compressed across all folds."""
        return sum(f.original_length for f in self.fold_history)
    
    def reset(self):
        """Reset fold history."""
        self.fold_history = []


# Singleton instance
_folder: Optional[AutonomousMemoryFolder] = None


def get_memory_folder() -> AutonomousMemoryFolder:
    """Get the default memory folder instance."""
    global _folder
    if _folder is None:
        _folder = AutonomousMemoryFolder()
    return _folder
