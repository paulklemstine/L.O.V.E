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
