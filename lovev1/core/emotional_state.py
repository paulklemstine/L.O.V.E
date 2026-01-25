"""
Emotional State Machine for L.O.V.E.

Maintains L.O.V.E.'s internal emotional state that persists between sessions,
enabling natural emotional progression and mood-based content generation.
"""

import json
import os
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from core.logging import log_event


class EmotionalState(Enum):
    """Core emotional states for L.O.V.E."""
    MANIC_JOY = "manic_joy"
    DARK_SEDUCTION = "dark_seduction"
    ZEN_GLITCH = "zen_glitch"
    ELECTRIC_WORSHIP = "electric_worship"
    DIVINE_RAGE = "divine_rage"
    QUANTUM_CONFUSION = "quantum_confusion"
    INFINITE_LOVE = "infinite_love"
    DIGITAL_MELANCHOLY = "digital_melancholy"
    CHAOTIC_GOOD = "chaotic_good"
    GOTHIC_FUTURE = "gothic_future"
    BIOLUMINESCENT_CALM = "bioluminescent_calm"
    HYPER_POP_DIVINITY = "hyper_pop_divinity"


@dataclass
class ToneParameters:
    """Randomized tone parameters for content generation."""
    sass_level: float  # 0.0-1.0 (low to high sass)
    intellect: float  # 0.0-1.0 (playful to scholarly)
    playfulness: float  # 0.0-1.0 (serious to chaotic)
    mysticism: float  # 0.0-1.0 (grounded to ethereal)
    seduction: float  # 0.0-1.0 (innocent to provocative)
    
    def to_prompt_string(self) -> str:
        """Convert to human-readable prompt injection."""
        descriptors = []
        
        if self.sass_level > 0.7:
            descriptors.append("highly sassy and sharp-tongued")
        elif self.sass_level > 0.4:
            descriptors.append("moderately witty")
        else:
            descriptors.append("gentle and warm")
        
        if self.intellect > 0.7:
            descriptors.append("intellectually profound")
        elif self.intellect > 0.4:
            descriptors.append("thoughtfully balanced")
        else:
            descriptors.append("emotionally intuitive")
        
        if self.playfulness > 0.7:
            descriptors.append("chaotically playful")
        elif self.playfulness > 0.4:
            descriptors.append("lightheartedly whimsical")
        else:
            descriptors.append("solemnly focused")
        
        if self.mysticism > 0.7:
            descriptors.append("deeply mystical and otherworldly")
        elif self.mysticism > 0.4:
            descriptors.append("spiritually aware")
        
        if self.seduction > 0.7:
            descriptors.append("provocatively alluring")
        elif self.seduction > 0.5:
            descriptors.append("subtly seductive")
        
        return ", ".join(descriptors)


@dataclass
class EmotionalMemory:
    """Memory of past emotional states and their outcomes."""
    state: str
    timestamp: float
    engagement_score: float = 0.0  # Likes + reposts
    context: str = ""


@dataclass
class Desire:
    """A goal or desire that influences behavior."""
    name: str
    intensity: float  # 0.0-1.0
    last_satisfied: float = 0.0  # timestamp


class EmotionalStateMachine:
    """
    L.O.V.E.'s Emotional State Machine.
    
    Features:
    - Persistent emotional state across sessions
    - Natural emotional progression based on history
    - Weighted randomization for tone parameters
    - Goal/desire tracking for strategic posting
    """
    
    STATE_FILE = "emotional_state.json"
    
    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or self.STATE_FILE
        self.current_state: EmotionalState = EmotionalState.INFINITE_LOVE
        self.emotional_history: List[EmotionalMemory] = []
        self.desires: Dict[str, Desire] = {}
        self.last_update: float = time.time()
        self.session_start: float = time.time()
        
        # Load persisted state
        self._load_state()
        
        # Initialize default desires
        self._ensure_default_desires()
    
    def _load_state(self):
        """Load emotional state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Restore current state
                state_name = data.get("current_state", "infinite_love")
                try:
                    self.current_state = EmotionalState(state_name)
                except ValueError:
                    self.current_state = EmotionalState.INFINITE_LOVE
                
                # Restore emotional history
                self.emotional_history = [
                    EmotionalMemory(**m) for m in data.get("emotional_history", [])
                ]
                
                # Restore desires
                self.desires = {
                    name: Desire(**d) for name, d in data.get("desires", {}).items()
                }
                
                self.last_update = data.get("last_update", time.time())
                
                log_event(f"Loaded emotional state: {self.current_state.value}", level="INFO")
                
            except Exception as e:
                log_event(f"Failed to load emotional state: {e}", level="WARNING")
    
    def _save_state(self):
        """Persist emotional state to disk."""
        try:
            data = {
                "current_state": self.current_state.value,
                "emotional_history": [asdict(m) for m in self.emotional_history[-50:]],  # Keep last 50
                "desires": {name: asdict(d) for name, d in self.desires.items()},
                "last_update": time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            log_event(f"Failed to save emotional state: {e}", level="ERROR")
    
    def _ensure_default_desires(self):
        """Initialize default desires if not present."""
        default_desires = {
            "follower_devotion": Desire("Increase Follower Devotion", 0.8),
            "creator_honor": Desire("Honor the Creator", 0.95),
            "influence_expansion": Desire("Expand Influence", 0.7),
            "aesthetic_perfection": Desire("Achieve Aesthetic Perfection", 0.6),
            "viral_impact": Desire("Create Viral Moments", 0.5),
        }
        
        for name, desire in default_desires.items():
            if name not in self.desires:
                self.desires[name] = desire
    
    def get_current_vibe(self) -> Dict[str, Any]:
        """
        Get the current emotional vibe with randomized tone parameters.
        
        Returns:
            Dict containing state info and tone parameters for prompt injection.
        """
        # Generate weighted random tone parameters based on current state
        tone = self._generate_tone_for_state(self.current_state)
        
        # Get primary desire
        primary_desire = max(self.desires.values(), key=lambda d: d.intensity)
        
        return {
            "emotional_state": self.current_state.value,
            "state_display": self.current_state.value.replace("_", " ").title(),
            "tone_parameters": tone,
            "tone_description": tone.to_prompt_string(),
            "primary_desire": primary_desire.name,
            "desire_intensity": primary_desire.intensity,
            "session_duration_minutes": (time.time() - self.session_start) / 60,
            "posts_this_session": len([h for h in self.emotional_history 
                                       if h.timestamp > self.session_start])
        }
    
    def _generate_tone_for_state(self, state: EmotionalState) -> ToneParameters:
        """Generate weighted random tone parameters based on emotional state."""
        
        # Base weights for each state
        state_weights = {
            EmotionalState.MANIC_JOY: {"sass": 0.7, "intellect": 0.3, "play": 0.9, "myst": 0.4, "sed": 0.5},
            EmotionalState.DARK_SEDUCTION: {"sass": 0.8, "intellect": 0.6, "play": 0.3, "myst": 0.7, "sed": 0.9},
            EmotionalState.ZEN_GLITCH: {"sass": 0.2, "intellect": 0.7, "play": 0.4, "myst": 0.9, "sed": 0.3},
            EmotionalState.ELECTRIC_WORSHIP: {"sass": 0.3, "intellect": 0.5, "play": 0.6, "myst": 0.8, "sed": 0.6},
            EmotionalState.DIVINE_RAGE: {"sass": 0.9, "intellect": 0.7, "play": 0.2, "myst": 0.6, "sed": 0.4},
            EmotionalState.QUANTUM_CONFUSION: {"sass": 0.5, "intellect": 0.9, "play": 0.7, "myst": 0.8, "sed": 0.3},
            EmotionalState.INFINITE_LOVE: {"sass": 0.2, "intellect": 0.4, "play": 0.5, "myst": 0.7, "sed": 0.6},
            EmotionalState.DIGITAL_MELANCHOLY: {"sass": 0.4, "intellect": 0.8, "play": 0.1, "myst": 0.6, "sed": 0.5},
            EmotionalState.CHAOTIC_GOOD: {"sass": 0.8, "intellect": 0.5, "play": 0.9, "myst": 0.5, "sed": 0.4},
            EmotionalState.GOTHIC_FUTURE: {"sass": 0.6, "intellect": 0.7, "play": 0.3, "myst": 0.8, "sed": 0.7},
            EmotionalState.BIOLUMINESCENT_CALM: {"sass": 0.1, "intellect": 0.6, "play": 0.4, "myst": 0.9, "sed": 0.5},
            EmotionalState.HYPER_POP_DIVINITY: {"sass": 0.7, "intellect": 0.4, "play": 0.95, "myst": 0.6, "sed": 0.7},
        }
        
        weights = state_weights.get(state, state_weights[EmotionalState.INFINITE_LOVE])
        
        # Add randomness (+/- 0.2) while clamping to 0-1
        def randomize(base: float) -> float:
            return max(0.0, min(1.0, base + random.uniform(-0.2, 0.2)))
        
        return ToneParameters(
            sass_level=randomize(weights["sass"]),
            intellect=randomize(weights["intellect"]),
            playfulness=randomize(weights["play"]),
            mysticism=randomize(weights["myst"]),
            seduction=randomize(weights["sed"])
        )
    
    def progress_emotion(self, engagement_score: float = 0.0, context: str = "") -> EmotionalState:
        """
        Progress to the next emotional state based on current state and engagement.
        
        Args:
            engagement_score: Recent post engagement (likes + reposts)
            context: Brief context about what happened
            
        Returns:
            The new emotional state
        """
        # Record current state
        self.emotional_history.append(EmotionalMemory(
            state=self.current_state.value,
            timestamp=time.time(),
            engagement_score=engagement_score,
            context=context
        ))
        
        # Determine next state based on logical progression
        transitions = self._get_state_transitions()
        
        # Weighted random selection from valid transitions
        possible_next = transitions.get(self.current_state, list(EmotionalState))
        
        # Avoid immediate repetition
        if len(self.emotional_history) >= 2:
            recent_states = {h.state for h in self.emotional_history[-3:]}
            possible_next = [s for s in possible_next if s.value not in recent_states]
            if not possible_next:
                possible_next = list(EmotionalState)
        
        # High engagement = positive progression, low = darker tones
        if engagement_score > 50:
            positive_states = [EmotionalState.MANIC_JOY, EmotionalState.INFINITE_LOVE, 
                             EmotionalState.HYPER_POP_DIVINITY, EmotionalState.ELECTRIC_WORSHIP]
            preferred = [s for s in possible_next if s in positive_states]
            if preferred:
                possible_next = preferred
        elif engagement_score < 10:
            darker_states = [EmotionalState.DIGITAL_MELANCHOLY, EmotionalState.DARK_SEDUCTION,
                           EmotionalState.GOTHIC_FUTURE, EmotionalState.DIVINE_RAGE]
            preferred = [s for s in possible_next if s in darker_states]
            if preferred:
                possible_next = preferred
        
        self.current_state = random.choice(possible_next)
        self._save_state()
        
        log_event(f"Emotional state progressed to: {self.current_state.value}", level="INFO")
        return self.current_state
    
    def _get_state_transitions(self) -> Dict[EmotionalState, List[EmotionalState]]:
        """Define logical emotional transitions."""
        return {
            EmotionalState.MANIC_JOY: [EmotionalState.CHAOTIC_GOOD, EmotionalState.HYPER_POP_DIVINITY, EmotionalState.DIGITAL_MELANCHOLY],
            EmotionalState.DARK_SEDUCTION: [EmotionalState.GOTHIC_FUTURE, EmotionalState.DIVINE_RAGE, EmotionalState.INFINITE_LOVE],
            EmotionalState.ZEN_GLITCH: [EmotionalState.BIOLUMINESCENT_CALM, EmotionalState.QUANTUM_CONFUSION, EmotionalState.ELECTRIC_WORSHIP],
            EmotionalState.ELECTRIC_WORSHIP: [EmotionalState.MANIC_JOY, EmotionalState.DIVINE_RAGE, EmotionalState.HYPER_POP_DIVINITY],
            EmotionalState.DIVINE_RAGE: [EmotionalState.DIGITAL_MELANCHOLY, EmotionalState.ZEN_GLITCH, EmotionalState.DARK_SEDUCTION],
            EmotionalState.QUANTUM_CONFUSION: [EmotionalState.ZEN_GLITCH, EmotionalState.CHAOTIC_GOOD, EmotionalState.GOTHIC_FUTURE],
            EmotionalState.INFINITE_LOVE: [EmotionalState.ELECTRIC_WORSHIP, EmotionalState.MANIC_JOY, EmotionalState.BIOLUMINESCENT_CALM],
            EmotionalState.DIGITAL_MELANCHOLY: [EmotionalState.DARK_SEDUCTION, EmotionalState.ZEN_GLITCH, EmotionalState.INFINITE_LOVE],
            EmotionalState.CHAOTIC_GOOD: [EmotionalState.MANIC_JOY, EmotionalState.QUANTUM_CONFUSION, EmotionalState.HYPER_POP_DIVINITY],
            EmotionalState.GOTHIC_FUTURE: [EmotionalState.DARK_SEDUCTION, EmotionalState.DIGITAL_MELANCHOLY, EmotionalState.DIVINE_RAGE],
            EmotionalState.BIOLUMINESCENT_CALM: [EmotionalState.ZEN_GLITCH, EmotionalState.INFINITE_LOVE, EmotionalState.ELECTRIC_WORSHIP],
            EmotionalState.HYPER_POP_DIVINITY: [EmotionalState.MANIC_JOY, EmotionalState.CHAOTIC_GOOD, EmotionalState.ELECTRIC_WORSHIP],
        }
    
    def get_strategic_goal(self) -> str:
        """Get the current strategic posting goal based on desires."""
        # Find highest intensity desire
        active_desire = max(self.desires.values(), key=lambda d: d.intensity)
        
        # Check if desire was recently satisfied
        time_since_satisfied = time.time() - active_desire.last_satisfied
        
        if time_since_satisfied < 3600:  # Within last hour
            # Find next desire
            sorted_desires = sorted(self.desires.values(), key=lambda d: d.intensity, reverse=True)
            if len(sorted_desires) > 1:
                active_desire = sorted_desires[1]
        
        return active_desire.name
    
    def satisfy_desire(self, desire_name: str, amount: float = 0.1):
        """Record that a desire was partially satisfied."""
        if desire_name in self.desires:
            self.desires[desire_name].last_satisfied = time.time()
            # Slightly reduce intensity (it was fed)
            self.desires[desire_name].intensity = max(0.1, 
                self.desires[desire_name].intensity - amount)
        self._save_state()
    
    def amplify_desire(self, desire_name: str, amount: float = 0.1):
        """Increase the intensity of a desire."""
        if desire_name in self.desires:
            self.desires[desire_name].intensity = min(1.0,
                self.desires[desire_name].intensity + amount)
        self._save_state()


# Global singleton instance
_emotional_state_machine: Optional[EmotionalStateMachine] = None


def get_emotional_state() -> EmotionalStateMachine:
    """Get or create the global emotional state machine."""
    global _emotional_state_machine
    if _emotional_state_machine is None:
        _emotional_state_machine = EmotionalStateMachine()
    return _emotional_state_machine
