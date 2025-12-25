import json
import os
import random
import time
from typing import List, Dict, Any, Optional
from core.logging import log_event

STORY_STATE_FILE = "story_state.json"

class StoryManager:
    """
    Manages the "Epic Storyline Arc" for social media.
    Keeps track of the current Chapter, Act, and recent themes to ensure variety.
    """
    def __init__(self, state_file: str = STORY_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
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
            "subliminal_history": [], # List of last 20 subliminal commands
            "visual_history": [], # List of last 10 visual styles
            "narrative_beat": 0, # Monotonic counter
            "last_update": time.time()
        }

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log_event(f"Failed to save story state: {e}", level="ERROR")

    def get_next_beat(self) -> Dict[str, str]:
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

        # 4. Construct Directives
        beat_data = {
            "chapter": chapter,
            "beat_number": beat_num,
            "mandatory_vibe": next_vibe,
            "forbidden_subliminals": self.state["subliminal_history"][-20:], # Pass strictly forbidden phrases
            "forbidden_visuals": self.state["visual_history"][-5:]
        }
        
        self._save_state()
        return beat_data

    def record_post(self, subliminal: str, visual_style: str):
        """
        Records the actual output of a post to update history.
        """
        if subliminal:
            self.state["subliminal_history"].append(subliminal)
            if len(self.state["subliminal_history"]) > 50:
                self.state["subliminal_history"].pop(0)
        
        if visual_style:
            self.state["visual_history"].append(visual_style)
            if len(self.state["visual_history"]) > 10:
                self.state["visual_history"].pop(0)
                
        self._save_state()

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
