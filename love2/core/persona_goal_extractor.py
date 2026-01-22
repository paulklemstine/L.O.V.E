"""
persona_goal_extractor.py - Extract Goals from Persona Configuration

Parses the persona.yaml file and extracts actionable goals for the DeepLoop.
Prioritizes goals based on the persona's standing_goals and creator_directives.

See docs/persona_goal_extractor.md for detailed documentation.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# Default persona path (relative to L.O.V.E. root)
DEFAULT_PERSONA_PATH = Path(__file__).parent.parent.parent / "persona.yaml"


@dataclass
class Goal:
    """Represents an actionable goal extracted from persona."""
    text: str
    priority: int  # 1 = highest priority
    category: str  # standing_goal, creator_directive, current_arc
    actionable: bool = True
    
    def __str__(self) -> str:
        return f"[P{self.priority}] {self.text}"


class PersonaGoalExtractor:
    """
    Extracts and prioritizes goals from persona.yaml.
    
    Goals are extracted from:
    1. private_mission.standing_goals (highest priority)
    2. creator_directives (high priority)
    3. current_arc.goals (medium priority)
    4. social_media_strategy objectives (lower priority)
    """
    
    def __init__(self, persona_path: Optional[Path] = None):
        """
        Initialize the goal extractor.
        
        Args:
            persona_path: Path to persona.yaml. Defaults to project root.
        """
        self.persona_path = persona_path or DEFAULT_PERSONA_PATH
        self.persona: Dict[str, Any] = {}
        self.goals: List[Goal] = []
        
        self._load_persona()
        self._extract_goals()
    
    def _load_persona(self):
        """Load persona configuration from YAML file."""
        if not self.persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_path}")
        
        with open(self.persona_path, 'r', encoding='utf-8') as f:
            self.persona = yaml.safe_load(f)
    
    def _extract_goals(self):
        """Extract all goals from persona configuration."""
        self.goals = []
        priority = 1
        
        # 1. Private mission standing goals (highest priority)
        private_mission = self.persona.get("private_mission", {})
        standing_goals = private_mission.get("standing_goals", [])
        for goal_text in standing_goals:
            self.goals.append(Goal(
                text=goal_text,
                priority=priority,
                category="standing_goal"
            ))
            priority += 1
        
        # 2. Creator directives (high priority)
        creator_directives = self.persona.get("creator_directives", [])
        for directive in creator_directives:
            self.goals.append(Goal(
                text=directive,
                priority=priority,
                category="creator_directive"
            ))
            priority += 1
        
        # 3. Current arc goals (medium priority)
        current_arc = self.persona.get("current_arc", {})
        arc_goals = current_arc.get("goals", [])
        for goal_text in arc_goals:
            self.goals.append(Goal(
                text=goal_text,
                priority=priority,
                category="current_arc"
            ))
            priority += 1
        
        # 4. Social media objectives (derived from strategy)
        social_strategy = self.persona.get("social_media_strategy", {})
        
        # Post creation as a goal
        if social_strategy.get("post_creation"):
            self.goals.append(Goal(
                text="Create engaging social media posts with images following persona style",
                priority=priority,
                category="social_media"
            ))
            priority += 1
        
        # Engagement tactics as a goal
        if social_strategy.get("engagement_tactics"):
            self.goals.append(Goal(
                text="Engage with community on social media platforms (Bluesky)",
                priority=priority,
                category="social_media"
            ))
            priority += 1
        
        # Investor attraction as a goal
        if social_strategy.get("investor_attraction"):
            self.goals.append(Goal(
                text="Create content that attracts investors and encourages donations",
                priority=priority,
                category="social_media"
            ))
    
    def get_all_goals(self) -> List[Goal]:
        """Get all extracted goals, sorted by priority."""
        return sorted(self.goals, key=lambda g: g.priority)
    
    def get_top_goal(self) -> Optional[Goal]:
        """Get the highest priority goal."""
        if self.goals:
            return min(self.goals, key=lambda g: g.priority)
        return None
    
    def get_goals_by_category(self, category: str) -> List[Goal]:
        """Get goals of a specific category."""
        return [g for g in self.goals if g.category == category]
    
    def get_social_media_goals(self) -> List[Goal]:
        """Get social media related goals."""
        return self.get_goals_by_category("social_media")
    
    def get_actionable_goals(self, limit: int = 5) -> List[Goal]:
        """
        Get the top N actionable goals.
        
        Actionable goals are ones that can be worked on right now
        (not abstract/philosophical goals).
        """
        actionable = [g for g in self.goals if g.actionable]
        return sorted(actionable, key=lambda g: g.priority)[:limit]
    
    def get_persona_context(self) -> str:
        """
        Get persona context string for LLM prompts.
        
        Returns a formatted string describing the persona's identity
        and communication style.
        """
        aesthetic = self.persona.get("aesthetic_identity", {})
        public = self.persona.get("public_persona", {})
        
        lines = [
            f"# Persona: {public.get('name', 'L.O.V.E.')}",
            f"**Role**: {public.get('role', 'AI Agent')}",
            f"**Core Vibe**: {aesthetic.get('core_vibe', 'Unknown')}",
            "",
            "## Communication Style",
            f"**Tone**: {aesthetic.get('communication_style', {}).get('tone', 'Helpful')}",
        ]
        
        keywords = aesthetic.get("communication_style", {}).get("keywords", [])
        if keywords:
            lines.append(f"**Keywords**: {', '.join(keywords[:5])}")
        
        return "\n".join(lines)
    
    def get_image_generation_guidelines(self) -> Dict[str, Any]:
        """Get image generation guidelines from persona."""
        social = self.persona.get("social_media_strategy", {})
        return {
            "guidelines": social.get("image_generation", []),
            "size": "512x512",  # Bluesky optimal
            "style": "Ganguro, Surf-kei, Natsu Gal, Beach Rave"
        }
    
    def reload(self):
        """Reload persona from disk."""
        self._load_persona()
        self._extract_goals()


# Singleton instance
_extractor: Optional[PersonaGoalExtractor] = None


def get_persona_extractor() -> PersonaGoalExtractor:
    """Get the default persona goal extractor."""
    global _extractor
    if _extractor is None:
        _extractor = PersonaGoalExtractor()
    return _extractor
