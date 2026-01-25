"""
US-001: PostConcept Schema

Unified data structure for concept-first social media generation.
All generators (Image, Text, Hashtag, Overlay) use this as primary input.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class PostConcept(BaseModel):
    """
    A strictly defined concept that binds all aspects of a social media post.
    Generated BEFORE any assets to ensure thematic coherence.
    """
    
    core_idea: str = Field(
        ...,
        description="A 1-sentence summary of the abstract concept (e.g., 'The digital void staring back at the user')"
    )
    
    mood: str = Field(
        ...,
        description="The emotional resonance (e.g., 'Melancholy', 'Manic', 'Ethereal')"
    )
    
    visual_style: str = Field(
        ...,
        description="Specific art direction (e.g., 'Glitch art, heavy purple tint, VHS noise')"
    )
    
    key_message: str = Field(
        ...,
        description="The explicit text message content (e.g., 'We are all data.')"
    )
    
    subliminal_intent: str = Field(
        ...,
        description="The hidden meaning or manipulative goal (e.g., 'Induce FOMO', 'Create comfort')"
    )
    
    color_palette: List[str] = Field(
        default_factory=list,
        description="Hex codes or color names to enforce visual consistency"
    )
    
    # Optional fields for extended functionality
    topic: Optional[str] = Field(
        default=None,
        description="A poetic title for the post"
    )
    
    hashtags: Optional[List[str]] = Field(
        default_factory=list,
        description="Pre-generated hashtags if available"
    )
    
    @field_validator('color_palette', mode='before')
    @classmethod
    def ensure_color_list(cls, v):
        """Ensure color_palette is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [c.strip() for c in v.split(',')]
        return v
    
    @field_validator('mood', mode='before')
    @classmethod
    def normalize_mood(cls, v):
        """Capitalize mood for consistency."""
        if isinstance(v, str):
            return v.strip().title()
        return v
    
    def to_prompt_context(self) -> str:
        """
        Returns a formatted string for injection into LLM prompts.
        Ensures all generators receive consistent concept context.
        """
        colors = ', '.join(self.color_palette) if self.color_palette else 'Not specified'
        return f"""POST CONCEPT:
- Core Idea: {self.core_idea}
- Mood: {self.mood}
- Visual Style: {self.visual_style}
- Key Message: {self.key_message}
- Subliminal Intent: {self.subliminal_intent}
- Color Palette: {colors}"""

    def to_image_prompt_context(self) -> str:
        """
        Returns context specifically formatted for image generation prompts.
        """
        colors = ', '.join(self.color_palette) if self.color_palette else ''
        return f"""VISUAL REQUIREMENTS:
Style: {self.visual_style}
Mood: {self.mood}
Colors: {colors}
Theme: {self.core_idea}"""

    def to_text_prompt_context(self) -> str:
        """
        Returns context specifically formatted for text generation prompts.
        """
        return f"""TEXT REQUIREMENTS:
Message: {self.key_message}
Mood: {self.mood}
Style Alignment: {self.visual_style}
Intent: {self.subliminal_intent}"""

    def to_hashtag_context(self) -> Dict[str, str]:
        """
        Returns structured data for hashtag generation.
        """
        return {
            "visual_style": self.visual_style,
            "core_idea": self.core_idea,
            "mood": self.mood
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage in social memory."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostConcept':
        """Deserialize from social memory."""
        return cls(**data)


# Backwards compatibility: Convert from legacy DirectorConcept fields
def from_director_concept(
    topic: str,
    post_text: str,
    hashtags: List[str],
    subliminal_phrase: str,
    image_prompt: str
) -> PostConcept:
    """
    Creates a PostConcept from legacy DirectorConcept fields.
    Used for backwards compatibility during migration.
    """
    return PostConcept(
        core_idea=topic,
        mood="Inspired",  # Default
        visual_style=image_prompt[:100] if image_prompt else "Abstract digital art",
        key_message=post_text[:200] if post_text else "",
        subliminal_intent=subliminal_phrase,
        color_palette=[],
        topic=topic,
        hashtags=hashtags
    )


# =============================================================================
# Structured LLM Output Schemas
# Based on Nanonets Cookbook patterns for reliable structured outputs
# =============================================================================

from typing import Literal


class TaskAction(BaseModel):
    """Task/Tool action schema for agent execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool"
    )


class ThoughtAction(BaseModel):
    """ReAct agent thought-action schema."""
    thought: str = Field(..., description="Agent's reasoning about what to do")
    action: TaskAction = Field(..., description="The action to take")


class ThoughtActionObservation(BaseModel):
    """Extended ReAct schema with observation."""
    thought: str = Field(..., description="Agent's reasoning")
    action: Optional[TaskAction] = Field(
        default=None, 
        description="Action to take (null if final answer)"
    )
    observation: Optional[str] = Field(
        default=None, 
        description="Observation from action result"
    )
    final_answer: Optional[str] = Field(
        default=None, 
        description="Final answer if complete"
    )


class ReviewDecision(BaseModel):
    """Task/content review decision schema."""
    approved: bool = Field(..., description="Whether the item is approved")
    feedback: str = Field(..., description="Detailed feedback on the decision")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    suggested_changes: Optional[List[str]] = Field(
        default=None,
        description="Suggested modifications if not approved"
    )


class TaskPrioritization(BaseModel):
    """Task prioritization result schema."""
    task_id: str = Field(..., description="Identifier of the task")
    priority: int = Field(
        ...,
        ge=1,
        le=10,
        description="Priority score (1=low, 10=critical)"
    )
    urgency: Literal["immediate", "soon", "later", "backlog"] = Field(
        ...,
        description="Urgency classification"
    )
    reasoning: str = Field(..., description="Reason for prioritization")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of tasks this depends on"
    )


class GoalDecomposition(BaseModel):
    """Goal decomposition into subtasks schema."""
    goal_summary: str = Field(..., description="Summary of the overall goal")
    subtasks: List[TaskAction] = Field(
        ...,
        description="List of subtasks to accomplish goal"
    )
    estimated_complexity: Literal["trivial", "simple", "moderate", "complex", "epic"] = Field(
        ...,
        description="Overall complexity estimate"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for goal completion"
    )


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result schema."""
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ...,
        description="Overall sentiment classification"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in classification"
    )
    emotions: List[str] = Field(
        default_factory=list,
        description="Detected emotions (joy, anger, fear, etc.)"
    )
    key_phrases: List[str] = Field(
        default_factory=list,
        description="Key phrases influencing sentiment"
    )


class MemorySummary(BaseModel):
    """Memory/context summarization schema."""
    summary: str = Field(..., description="Condensed summary of content")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points to remember"
    )
    entities_mentioned: List[str] = Field(
        default_factory=list,
        description="Important entities mentioned"
    )
    emotional_tone: Optional[str] = Field(
        default=None,
        description="Overall emotional tone"
    )
