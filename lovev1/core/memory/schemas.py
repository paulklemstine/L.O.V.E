from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class KeyEvent(BaseModel):
    step: int
    action: str
    outcome: str

class EpisodicMemory(BaseModel):
    task_description: str = Field(default="")
    key_events: List[KeyEvent] = Field(default_factory=list)

class WorkingMemory(BaseModel):
    current_subgoal: str = Field(default="")
    pending_tasks: List[str] = Field(default_factory=list)
    active_variables: Dict[str, Any] = Field(default_factory=dict)

class ToolUsage(BaseModel):
    tool_name: str
    success_rate: float
    effective_params: Dict[str, Any]

class ToolMemory(BaseModel):
    tools_used: List[ToolUsage] = Field(default_factory=list)

class MemorySummary(BaseModel):
    content: str
    level: int
    source_ids: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    ipfs_cid: Optional[str] = None
    embedding: Optional[List[float]] = None


class KnowledgeNugget(BaseModel):
    """
    A compressed summary of a conversation thread (Story 2.2).
    
    Created when token count > 80% of limit, this nugget replaces
    the oldest 50% of messages in the active context while preserving
    critical directives and key insights.
    """
    content: str  # The summarized content
    source_message_count: int  # How many messages were compressed
    key_directives: List[str] = Field(default_factory=list)  # Critical instructions preserved
    topics: List[str] = Field(default_factory=list)  # Main topics covered
    created_at: float = Field(default_factory=lambda: __import__("time").time())
    ipfs_cid: Optional[str] = None  # For cold storage
    token_savings: int = 0  # Approximate tokens saved by this compression


# =============================================================================
# Story 2.1: The Ouroboros Memory Fold - WisdomEntry
# =============================================================================

class WisdomEntry(BaseModel):
    """
    A distilled lesson learned from operational experience (Story 2.1).
    
    These entries compress experience into wisdom (heuristics) that update
    the prompt context dynamically, enabling recursive improvement.
    
    The Morphic Principle: The output of one iteration forms the context
    for the next. Evolution is a spiral, not a straight line.
    """
    situation: str = Field(
        ..., 
        description="What was the context/problem? The circumstances that led to action."
    )
    action: str = Field(
        ..., 
        description="What action was taken? The specific decision or tool used."
    )
    outcome: str = Field(
        ..., 
        description="What was the result? Success, failure, or partial outcome."
    )
    principle: str = Field(
        ..., 
        description="What lesson was derived? The wisdom to apply in future situations."
    )
    confidence: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0,
        description="How confident are we in this wisdom? 0.0 to 1.0"
    )
    source: str = Field(
        default="experience",
        description="Where this wisdom came from: 'experience', 'failure', 'success', 'external'"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and retrieval (e.g., 'tool_usage', 'reasoning', 'error_handling')"
    )
    created_at: float = Field(
        default_factory=lambda: __import__("time").time()
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Semantic embedding for similarity-based retrieval"
    )
    
    def to_prompt_format(self) -> str:
        """Formats this wisdom entry for injection into a system prompt."""
        return (
            f"**Situation**: {self.situation}\n"
            f"**Action**: {self.action}\n"
            f"**Outcome**: {self.outcome}\n"
            f"**Principle**: {self.principle}"
        )
    
    def to_concise_format(self) -> str:
        """Formats as a single-line wisdom principle."""
        return f"When {self.situation.lower()}, {self.principle.lower()}"


