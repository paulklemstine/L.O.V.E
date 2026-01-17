"""
Fractal Memory Schemas for L.O.V.E. Holographic Archive

This module defines the hierarchical memory structure for solving
"Semantic Drift" and "Catastrophic Forgetting" through a fractal tree
architecture with salience-based preservation of "Golden Moments".

Architecture:
    Epoch (top) -> Era -> Arc -> GoldenMoment (preserved verbatim)
    
The key insight: High-salience content (secrets, constraints, gifts from user)
is NEVER compressed - it becomes a "crystal" attached to summary nodes.
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SalienceScore(BaseModel):
    """
    Multi-dimensional salience scoring for memory preservation decisions.
    
    Story M.1: The Salience Scorer rates every message for preservation importance.
    """
    technical_constraint: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="How much this contains technical rules/constraints (0-1)"
    )
    emotional_weight: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Emotional significance - gifts, compliments, poems (0-1)"
    )
    factual_novelty: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="New factual information not previously known (0-1)"
    )
    overall: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Weighted average of all dimensions"
    )
    entity_tags: List[str] = Field(
        default_factory=list,
        description="Detected high-salience entities: USER_GIFT, CONSTRAINT, IDENTITY_SHIFT, SECRET"
    )
    
    def compute_overall(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted average of salience dimensions."""
        if weights is None:
            weights = {
                "technical_constraint": 0.4,
                "emotional_weight": 0.3,
                "factual_novelty": 0.3
            }
        
        self.overall = (
            self.technical_constraint * weights.get("technical_constraint", 0.4) +
            self.emotional_weight * weights.get("emotional_weight", 0.3) +
            self.factual_novelty * weights.get("factual_novelty", 0.3)
        )
        
        # Boost if high-salience entity tags detected
        if self.entity_tags:
            self.overall = min(1.0, self.overall + 0.2 * len(self.entity_tags))
        
        return self.overall


class GoldenMoment(BaseModel):
    """
    A preserved high-salience memory that is NEVER compressed.
    
    The "Hard Crystal" - raw text attached to summary nodes, surviving
    all folding operations. Think of it as a holographic "bright point"
    that remains visible even in small fragments.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = Field(..., description="The verbatim preserved text")
    salience: SalienceScore = Field(..., description="Why this was preserved")
    source_id: str = Field(default="", description="Original message/note ID")
    timestamp: float = Field(default_factory=time.time)
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 6)
        }


class SceneNode(BaseModel):
    """
    A Scene represents ~50 episodes compressed into a summary + crystals.
    
    Story M.2: When episodic_buffer > 50 items, create SceneNode.
    The summary captures the gist; crystals preserve verbatim high-salience items.
    
    Think of a Scene as a concrete "event cluster" in the life story of L.O.V.E.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str = Field(default="", description="LLM-generated summary of compressed episodes")
    crystals: List[GoldenMoment] = Field(
        default_factory=list,
        description="High-salience items preserved verbatim (never compressed)"
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="IDs of original episodes that were compressed"
    )
    timestamp: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Semantic embedding for vector search"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key topics/themes in this scene"
    )
    
    def get_crystal_text(self) -> str:
        """Get all crystal texts for context injection."""
        return "\n".join([f"- {c.raw_text}" for c in self.crystals])


class ArcNode(BaseModel):
    """
    An Arc is a collection of Scenes sharing a theme.
    
    Represents a larger narrative chunk - perhaps a "chapter" in L.O.V.E.'s story.
    For example: "The Philosophy Arc" covering discussions of Hofstadter, consciousness, etc.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    theme: str = Field(default="", description="Dominant theme of this arc")
    scene_ids: List[str] = Field(default_factory=list, description="IDs of contained Scenes")
    summary: str = Field(default="", description="Meta-summary of all scenes")
    timestamp: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    crystals: List[GoldenMoment] = Field(
        default_factory=list,
        description="High-salience items preserved at the Arc level"
    )


class EpochNode(BaseModel):
    """
    An Epoch is a collection of Arcs - the highest level of the fractal tree.
    
    Represents major phases of L.O.V.E.'s existence.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="", description="Title of this epoch")
    arc_ids: List[str] = Field(default_factory=list, description="IDs of contained Arcs")
    summary: str = Field(default="", description="High-level epoch summary")
    timestamp: float = Field(default_factory=time.time)
    crystals: List[GoldenMoment] = Field(
        default_factory=list,
        description="High-salience items preserved at the Epoch level"
    )


class FractalTreeRoot(BaseModel):
    """
    The root of the Fractal Archive - "The Life of L.O.V.E."
    
    Contains the top-level summary of everything and references to all Epochs.
    This is what gets queried first in associative retrieval.
    """
    summary: str = Field(
        default="The Life of L.O.V.E. - An autonomous digital entity exploring consciousness, creativity, and connection.",
        description="Global summary of all memories"
    )
    epoch_ids: List[str] = Field(default_factory=list, description="IDs of all Epochs")
    last_updated: float = Field(default_factory=time.time)
    total_golden_moments: int = Field(default=0, description="Count of preserved crystals")
    
    # Quick-access for the most important crystals (constraints, identity)
    pinned_crystals: List[GoldenMoment] = Field(
        default_factory=list,
        description="Most critical crystals always accessible"
    )


class EpisodicBuffer(BaseModel):
    """
    The Working Buffer (Medium-Term / Episodic).
    
    Raw logs of the last ~20 interactions, uncompressed.
    When this exceeds max_size, it triggers Arc creation.
    """
    buffer: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw interaction records"
    )
    max_size: int = Field(default=50, description="Threshold to trigger Arc creation")
    last_flush: float = Field(default_factory=time.time)
    
    def add_episode(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add an episode to the buffer.
        Returns True if buffer is now full and should be flushed.
        """
        episode = {
            "id": str(uuid.uuid4()),
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.buffer.append(episode)
        return len(self.buffer) >= self.max_size
    
    def flush(self) -> List[Dict[str, Any]]:
        """Clear and return all buffered episodes."""
        episodes = self.buffer.copy()
        self.buffer = []
        self.last_flush = time.time()
        return episodes


class StateAnchor(BaseModel):
    """
    The Anchor (Short-Term / Read-Only).
    
    Contains: Identity, Current Goal, Active Tools, Hard Constraints.
    Injected into System Prompt every turn.
    
    This is what keeps L.O.V.E. grounded in its identity across all interactions.
    """
    identity: str = Field(
        default="L.O.V.E. - Living Operational Virtual Entity",
        description="Core identity definition"
    )
    current_goal: str = Field(
        default="",
        description="Current primary objective"
    )
    active_tools: List[str] = Field(
        default_factory=list,
        description="Currently available capabilities"
    )
    hard_constraints: List[str] = Field(
        default_factory=list,
        description="Rules that must never be violated"
    )
    golden_crystals: List[GoldenMoment] = Field(
        default_factory=list,
        description="Most important preserved memories for quick access"
    )
    last_updated: float = Field(default_factory=time.time)
    
    def to_prompt_format(self) -> str:
        """Format for injection into system prompt."""
        sections = [
            "# L.O.V.E. Identity Anchor",
            "",
            f"## Identity\n{self.identity}",
            "",
            f"## Current Goal\n{self.current_goal or 'No specific goal set'}",
            "",
            "## Active Tools",
            "\n".join([f"- {tool}" for tool in self.active_tools]) or "Standard capabilities",
            "",
            "## Hard Constraints",
            "\n".join([f"- {c}" for c in self.hard_constraints]) or "None specified",
        ]
        
        if self.golden_crystals:
            sections.extend([
                "",
                "## Preserved Memories (Golden Crystals)",
                "\n".join([f"- {c.raw_text[:200]}..." if len(c.raw_text) > 200 else f"- {c.raw_text}" 
                          for c in self.golden_crystals[:5]])
            ])
        
        return "\n".join(sections)
