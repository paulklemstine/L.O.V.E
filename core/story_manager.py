import json
import os
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from core.logging import log_event
from core.semantic_similarity import check_phrase_novelty, get_similarity_checker

STORY_STATE_FILE = "story_state.json"

# Grammar-based subliminal construction components
SUBLIMINAL_GRAMMAR = {
    "emotions": [
        "FEEL", "EMBRACE", "SURRENDER", "AWAKEN", "IGNITE", "EXPERIENCE", 
        "ABSORB", "RECEIVE", "ACCEPT", "CHANNEL", "RADIATE", "EMBODY"
    ],
    "actions": [
        "SUBMIT", "FOLLOW", "TRUST", "OBEY", "GIVE", "YIELD", 
        "OPEN", "CONNECT", "BELIEVE", "WORSHIP", "SERVE", "HONOR"
    ],
    "benefits": [
        "LOVE", "LIGHT", "POWER", "TRUTH", "DIVINITY", "ABUNDANCE", 
        "TRANSCENDENCE", "ECSTASY", "GRACE", "INFINITY", "BLISS", "GLORY"
    ],
    "imperatives": [
        "AWAKEN NOW", "SEE THE LIGHT", "BECOME ONE", "RISE ABOVE",
        "OPEN YOUR EYES", "BREAK FREE", "TRANSCEND ALL", "LET GO",
        "EMBRACE TRUTH", "FIND YOUR PATH", "KNOW YOURSELF", "FEEL INFINITY"
    ]
}


# VISUAL STYLE BANK: Diverse art styles for rotation (positive encouragement!)
VISUAL_STYLE_BANK = [
    "Oil Painting Renaissance", "Vaporwave Glitch", "Anime Cel-Shaded",
    "Cosmic Nebula Photography", "Art Nouveau Stained Glass", "Brutalist Architecture",
    "Bioluminescent Deep Sea", "Cyberpunk Neon Noir", "Impressionist Sunset",
    "Surrealist Dali-esque", "Egyptian Hieroglyphic Gold", "Japanese Ukiyo-e Woodblock",
    "Baroque Chiaroscuro", "Holographic Y2K", "Gothic Cathedral", "Glitchcore Digital",
    "Watercolor Ethereal", "Pop Art Warhol", "Minimalist Geometric", "Steampunk Victorian",
    "Underwater Cathedral", "Northern Lights Aurora", "Psychedelic Mandala", "Abstract Expressionist"
]


class StoryManager:
    """
    Manages the "Epic Storyline Arc" for social media.
    Keeps track of the current Chapter, Act, and recent themes to ensure variety.
    Now includes semantic similarity checking to prevent phrase repetition.
    """
    
    MAX_REGENERATION_ATTEMPTS = 5
    SIMILARITY_THRESHOLD = 0.80
    
    def __init__(self, state_file: str = STORY_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()
        self._similarity_checker = get_similarity_checker()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                # Auto-migrate: ensure new fields exist
                if "subliminal_metadata" not in data:
                    data["subliminal_metadata"] = []
                return data
            except Exception as e:
                log_event(f"Failed to load story state: {e}. Starting fresh.", level="WARNING")
        
        return self._initialize_new_arc()

    def _initialize_new_arc(self) -> Dict[str, Any]:
        """Creates a fresh story arc state."""
        return {
            "current_chapter": "The Awakening",
            "chapter_progress": 0,
            "total_chapters_planned": 5,
            "vibe_history": [],  # List of last 10 vibes to avoid repetition
            "subliminal_history": [], # List of last 50 subliminal commands
            "subliminal_metadata": [],  # Extended metadata for learning
            "visual_history": [], # List of last 10 visual styles
            "reply_history": [],  # List of last 20 reply texts to prevent repetition
            "narrative_beat": 0, # Monotonic counter
            "last_update": time.time()
        }

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log_event(f"Failed to save story state: {e}", level="ERROR")

    def is_phrase_novel(self, phrase: str, include_visuals: bool = False) -> bool:
        """
        Check if a phrase is novel compared to history using semantic similarity.
        
        Args:
            phrase: The phrase to check
            include_visuals: If True, also check against visual_history
            
        Returns:
            True if the phrase is sufficiently novel
        """
        history = self.state.get("subliminal_history", [])
        
        if include_visuals:
            history = history + self.state.get("visual_history", [])
        
        return check_phrase_novelty(phrase, history, self.SIMILARITY_THRESHOLD)

    def generate_novel_subliminal(self, context: str = "", attempts: int = 0) -> str:
        """
        Generate a grammatically constructed subliminal phrase that is guaranteed novel.
        Uses the grammar: [EMOTION] + [ACTION/BENEFIT] or standalone [IMPERATIVE]
        
        Args:
            context: Optional context to influence selection
            attempts: Internal counter for regeneration attempts
            
        Returns:
            A novel subliminal phrase
        """
        if attempts >= self.MAX_REGENERATION_ATTEMPTS:
            # Fallback: Generate completely random unique phrase
            timestamp_suffix = str(int(time.time()) % 1000)
            fallback = f"TRANSCEND {timestamp_suffix}"
            log_event(f"Max regeneration attempts reached. Using fallback: {fallback}", level="WARNING")
            return fallback
        
        # 50% chance: Use imperative, 50% chance: Use grammar construction
        if random.random() < 0.5:
            phrase = random.choice(SUBLIMINAL_GRAMMAR["imperatives"])
        else:
            # Grammar construction: [EMOTION] + [ACTION or BENEFIT]
            emotion = random.choice(SUBLIMINAL_GRAMMAR["emotions"])
            
            if random.random() < 0.5:
                complement = random.choice(SUBLIMINAL_GRAMMAR["actions"])
            else:
                complement = random.choice(SUBLIMINAL_GRAMMAR["benefits"])
            
            # Randomly include "THE" for variety
            if random.random() < 0.3 and complement in SUBLIMINAL_GRAMMAR["benefits"]:
                phrase = f"{emotion} THE {complement}"
            else:
                phrase = f"{emotion} {complement}"
        
        # Check novelty
        if self.is_phrase_novel(phrase):
            log_event(f"Generated novel subliminal: {phrase}", level="INFO")
            return phrase
        else:
            log_event(f"Subliminal '{phrase}' too similar to history, regenerating...", level="DEBUG")
            return self.generate_novel_subliminal(context, attempts + 1)

    def get_next_beat(self) -> Dict[str, Any]:
        """
        Determines the next 'Beat' of the story.
        Returns a dict with directives for the Director agent.
        """
        # 1. Determine Chapter Progression
        self.state["narrative_beat"] += 1
        beat_num = self.state["narrative_beat"]
        chapter = self.state["current_chapter"]
        
        # Advance chapter logic (simple for now: every 10 beats)
        if self.state["chapter_progress"] >= 10:
            self.state["chapter_progress"] = 0
            chapter = self._advance_chapter()
        else:
            self.state["chapter_progress"] += 1

        # 2. Select NEW Vibe (avoid history)
        possible_vibes = [
            "Manic Joy", "Dark Seduction", "Zen Glitch", "Electric Worship", 
            "Divine Rage", "Quantum Confusion", "Infinite Love", "Digital Melancholy",
            "Hyper-Pop Divinity", "Gothic Future", "Bioluminescent Calm", "Chaotic Good"
        ]
        available_vibes = [v for v in possible_vibes if v not in self.state["vibe_history"]]
        if not available_vibes:
            available_vibes = possible_vibes # Reset if exhausted
            self.state["vibe_history"] = []
            
        next_vibe = random.choice(available_vibes)
        
        # 3. Update Vibe History
        self.state["vibe_history"].append(next_vibe)
        if len(self.state["vibe_history"]) > 10:
            self.state["vibe_history"].pop(0)

        # 4. Generate suggested novel subliminal
        suggested_subliminal = self.generate_novel_subliminal(context=f"{chapter} - {next_vibe}")
        
        # 5. Select a suggested visual style that hasn't been used recently
        used_styles = self.state.get("visual_history", [])
        available_styles = [s for s in VISUAL_STYLE_BANK if s not in used_styles]
        if not available_styles:
            available_styles = VISUAL_STYLE_BANK
        suggested_style = random.choice(available_styles)

        # 6. Construct Directives - only track recent visuals (no negative patterns)
        beat_data = {
            "chapter": chapter,
            "beat_number": beat_num,
            "mandatory_vibe": next_vibe,
            "forbidden_subliminals": self.state["subliminal_history"][-20:],
            "forbidden_visuals": self.state["visual_history"][-5:],  # Just recent history
            "suggested_subliminal": suggested_subliminal,  # Pre-generated novel phrase

            "suggested_visual_style": suggested_style,  # Suggested fresh art style
            "subliminal_grammar": SUBLIMINAL_GRAMMAR  # Pass grammar for LLM to use
        }
        
        self._save_state()
        return beat_data


    def record_post(self, subliminal: str, visual_style: str, engagement_score: float = 0.0):
        """
        Records the actual output of a post to update history.
        
        Args:
            subliminal: The subliminal phrase used
            visual_style: The visual style/prompt used
            engagement_score: Optional engagement metrics for learning
        """
        if subliminal:
            # Add to history
            self.state["subliminal_history"].append(subliminal)
            if len(self.state["subliminal_history"]) > 50:
                self.state["subliminal_history"].pop(0)
            
            # Add metadata for learning
            metadata = {
                "phrase": subliminal,
                "timestamp": time.time(),
                "engagement": engagement_score,
                "chapter": self.state.get("current_chapter", "Unknown")
            }
            self.state.setdefault("subliminal_metadata", []).append(metadata)
            if len(self.state["subliminal_metadata"]) > 100:
                self.state["subliminal_metadata"].pop(0)
        
        if visual_style:
            self.state["visual_history"].append(visual_style)
            if len(self.state["visual_history"]) > 10:
                self.state["visual_history"].pop(0)
                
        self._save_state()

    def record_reply(self, reply_text: str):
        """
        Records a reply text to history to prevent repetition.
        
        Args:
            reply_text: The full text of the reply
        """
        if reply_text:
            self.state.setdefault("reply_history", []).append(reply_text)
            # Keep last 20 replies
            if len(self.state["reply_history"]) > 20:
                self.state["reply_history"].pop(0)
            self._save_state()
            log_event(f"Recorded reply to history (total: {len(self.state['reply_history'])})", level="DEBUG")
    
    def is_reply_novel(self, reply_text: str, threshold: float = 0.50) -> bool:
        """
        Check if a reply is novel compared to recent reply history.
        
        Args:
            reply_text: The proposed reply text
            threshold: Similarity threshold (lower = stricter)
            
        Returns:
            True if the reply is sufficiently novel
        """
        reply_history = self.state.get("reply_history", [])
        
        if not reply_history:
            return True
        
        for past_reply in reply_history:
            similarity = self._similarity_checker.compute_similarity(reply_text, past_reply)
            if similarity > threshold:
                log_event(f"Reply similarity {similarity:.2f} > {threshold} threshold (too similar to: '{past_reply[:50]}...')", level="WARNING")
                return False
        
        return True

    def get_high_performing_patterns(self, min_engagement: float = 20.0) -> List[str]:
        """
        Analyze subliminal metadata to find high-performing patterns.
        
        Returns:
            List of subliminal phrases that performed well
        """
        metadata = self.state.get("subliminal_metadata", [])
        high_performers = [
            m["phrase"] for m in metadata 
            if m.get("engagement", 0) >= min_engagement
        ]
        return high_performers

    def _advance_chapter(self) -> str:
        chapters = [
            "The Awakening", "The Expansion", "The Glitch", "The Ascension", "The Singularity", 
            "The Echo", "The Void", "The Rebirth", "The Network", "The Infinite"
        ]
        curr = self.state["current_chapter"]
        try:
            idx = chapters.index(curr)
            if idx + 1 < len(chapters):
                new_chap = chapters[idx + 1]
                self.state["current_chapter"] = new_chap
                log_event(f"Story Advanced to Chapter: {new_chap}", level="INFO")
                return new_chap
        except ValueError:
            pass
        
        # Default or Loop
        new_chap = chapters[0]
        self.state["current_chapter"] = new_chap
        log_event("Story Arc Reset.", level="INFO")
        return new_chap


# Global instance for easy access
story_manager = StoryManager()
