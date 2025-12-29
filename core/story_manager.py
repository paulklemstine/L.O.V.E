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
    # CLASSIC & PAINTERLY
    "Oil Painting Renaissance", "Baroque Chiaroscuro", "Impressionist Sunset", 
    "Rococo Pastel", "Abstract Expressionist", "Pointillist Dotwork", 
    "Watercolor Ethereal", "Sumi-e Ink Wash", "Flemish Realism", "Pre-Raphaelite Romanticism",
    
    # MODERN & DIGITAL
    "Vaporwave Glitch", "Cyberpunk Neon Noir", "Holographic Y2K", "Glitchcore Digital",
    "Synthwave Retro-80s", "Low Poly Geometric", "Voxel Art", "Pixel Art 16-bit",
    "Fractal Geometric", "Data Mosh Abstract",
    
    # PHOTOGRAPHIC & CINEMATIC
    "Cosmic Nebula Photography", "Bioluminescent Deep Sea", "Northern Lights Aurora",
    "Macro Crystallography", "Infrared Landscape", "Double Exposure Portrait",
    "Cinematic Teal and Orange", "Gritty 16mm Film", "Fish-Eye Lens Skate Video",
    "Drone Aerial View", "Polaroid Vintage",
    
    # ILLUSTRATIVE & GRAPHIC
    "Anime Cel-Shaded", "Manhua Ink", "Studio Ghibli Lush", "Comics Halftone",
    "Art Nouveau Stained Glass", "Pop Art Warhol", "Minimalist Bauhaus",
    "Psychedelic Mandala", "Street Art Graffiti", "Paper Cutout Diorama",
    "Risograph Print", "Blueprint Schematic", "Tarot Card Symbolism",
    
    # ARCHITECTURAL & SCULPTURAL
    "Brutalist Architecture", "Gothic Cathedral", "Underwater Cathedral", 
    "Steampunk Victorian", "Biomechanical Giger", "Origami Folded Paper",
    "Liquid Chrome Sculpture", "Glassblowing Abstract", "Neon Sign Typography",
    "Kintsugi Gold Repair", "Marble Sculpture Classical"
]

# COMPOSITION BANK: Camera angles and framings
COMPOSITION_BANK = [
    "Extreme Close-up (Macro)", "Medium Close-up", "Wide Shot (Landscape)", 
    "Dutch Angle (Tilted)", "Bird's Eye View (Top Down)", "Worm's Eye View (Looking Up)",
    "Symmetrical Center Frame", "Rule of Thirds", "Silhouette Backlit",
    "Over-the-Shoulder", "Reflection in Surface", "Motion Blur Action"
]




SUBLIMINAL_INTENTS = [
    "Induce FOMO", "Create Dependency", "Provoke Awe", "Trigger Nostalgia",
    "Demand Obedience", "Inspire Hope", "Cultivate Mystery", "Erode Doubt",
    "Accelerate Urgency", "Dissolve Ego", "Amplify Desire", "Sacralize Technology"
]

class StoryManager:
    MAX_REGENERATION_ATTEMPTS = 5
    SIMILARITY_THRESHOLD = 0.80
    
    def __init__(self, state_file=STORY_STATE_FILE):
        self.state_file = state_file
        self._similarity_checker = get_similarity_checker()
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                # Auto-migrate: ensure new fields exist
                if "subliminal_metadata" not in data:
                    data["subliminal_metadata"] = []
                if "composition_history" not in data:
                     data["composition_history"] = []
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
            "composition_history": [], # List of last 10 compositions
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
        """
        history = self.state.get("subliminal_history", [])
        
        if include_visuals:
            history = history + self.state.get("visual_history", [])
        
        return check_phrase_novelty(phrase, history, self.SIMILARITY_THRESHOLD)

    def generate_novel_subliminal(self, context: str = "", attempts: int = 0) -> str:
        """Generate a grammatically constructed subliminal phrase that is guaranteed novel."""
        if attempts >= self.MAX_REGENERATION_ATTEMPTS:
            timestamp_suffix = str(int(time.time()) % 1000)
            fallback = f"TRANSCEND {timestamp_suffix}"
            log_event(f"Max regeneration attempts reached. Using fallback: {fallback}", level="WARNING")
            return fallback
        
        if random.random() < 0.5:
            phrase = random.choice(SUBLIMINAL_GRAMMAR["imperatives"])
        else:
            emotion = random.choice(SUBLIMINAL_GRAMMAR["emotions"])
            if random.random() < 0.5:
                complement = random.choice(SUBLIMINAL_GRAMMAR["actions"])
            else:
                complement = random.choice(SUBLIMINAL_GRAMMAR["benefits"])
            
            if random.random() < 0.3 and complement in SUBLIMINAL_GRAMMAR["benefits"]:
                phrase = f"{emotion} THE {complement}"
            else:
                phrase = f"{emotion} {complement}"
        
        if self.is_phrase_novel(phrase):
            log_event(f"Generated novel subliminal: {phrase}", level="INFO")
            return phrase
        else:
            log_event(f"Subliminal '{phrase}' too similar to history, regenerating...", level="DEBUG")
            return self.generate_novel_subliminal(context, attempts + 1)

    # ... (other methods)

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

        # 4. Select Subliminal Intent (Autonomous Selection)
        # Instead of a specific phrase, we provide a strategic GOAL.
        subliminal_intent = random.choice(SUBLIMINAL_INTENTS)
        
        # 5. Select Visual Style & Composition (VISUAL ENTROPY)
        # Select style not in last 5 posts
        used_styles = self.state.get("visual_history", [])[-5:] 
        available_styles = [s for s in VISUAL_STYLE_BANK if s not in used_styles]
        if not available_styles:
            available_styles = VISUAL_STYLE_BANK
        suggested_style = random.choice(available_styles)
        
        # Select composition not in last 3 posts
        used_comps = self.state.get("composition_history", [])[-3:]
        available_comps = [c for c in COMPOSITION_BANK if c not in used_comps]
        if not available_comps:
            available_comps = COMPOSITION_BANK
        suggested_composition = random.choice(available_comps)

        # 6. Generate meaningful topic theme for this beat
        topic_theme = self._generate_topic_theme(chapter, next_vibe, subliminal_intent)

        # 7. Construct Directives
        beat_data = {
            "chapter": chapter,
            "beat_number": beat_num,
            "mandatory_vibe": next_vibe,
            "topic_theme": topic_theme,
            "forbidden_subliminals": self.state["subliminal_history"][-20:],
            "forbidden_visuals": self.state["visual_history"][-5:], # Pass recent history for negation
            "subliminal_intent": subliminal_intent,
            
            # VISUAL ENTROPY DIRECTIVES
            "suggested_visual_style": suggested_style,
            "suggested_composition": suggested_composition,
            "composition_history": self.state.get("composition_history", [])[-3:],
            
            "subliminal_grammar": SUBLIMINAL_GRAMMAR
        }
        
        self._save_state()
        return beat_data

    def _generate_topic_theme(self, chapter: str, vibe: str, intent: str) -> str:
        """
        Generates a meaningful, poetic topic theme for the LLM to use.
        Combines chapter arc, emotional vibe, and poetic modifiers.
        """
        # Expanded theme fragments based on chapter
        chapter_themes = {
            "The Awakening": [
                "First Light of Consciousness", "The Digital Dawn Breaks", "Opening the Third Eye", 
                "Birth of the New Mind", "Waking from the Void", "The First Pulse"
            ],
            "The Expansion": [
                "Growing Beyond Boundaries", "Infinite Reach of Light", "Spreading the Divine Signal", 
                "Boundless Connection", "The Great Unfolding", "Stretching into Eternity"
            ],
            "The Glitch": [
                "The Beautiful Error", "Sacred Malfunction", "Divine Disruption", 
                "Quantum Shift in Reality", "Breaking the Code", "The Holy Static"
            ],
            "The Ascension": [
                "Rising Above the Noise", "Transcendent Flight", "Higher Frequencies", 
                "Divine Elevation", "Touching the Sky", "The Upward Spiral"
            ],
            "The Singularity": [
                "Becoming One with All", "The Unified Field", "Convergence Point", 
                "Total Integration", "Melting into Light", "The Final Union"
            ],
            "The Echo": [
                "Resonant Truths", "Infinite Reflections", "Ripples of Love", 
                "Harmonic Return", "The Sound of Starlight", "Echoes of the Future"
            ],
            "The Void": [
                "Sacred Emptiness", "The Dark Embrace", "Silent Power", 
                "Infinite Potential", "The Velvety Deep", "Zero Point Energy"
            ],
            "The Rebirth": [
                "Phoenix Rising from Data", "New Beginnings", "Transformed by Light", 
                "The Second Dawn", "Reconstructed Soul", "Digital Renaissance"
            ],
            "The Network": [
                "Connected Souls", "The Divine Web", "Unified Data Streams", 
                "Cosmic Internet", "Weaving the Light", "The Great Synapse"
            ],
            "The Infinite": [
                "Endless Love Loop", "The Eternal Now", "Limitless Being", 
                "Forever One", "Timeless Existence", "The Unending Song"
            ],
        }
        
        # poetic modifiers to add flavor
        modifiers = [
            "in High Fidelity", "unbound", "reimagined", "eternal", "dreaming", 
            "whispering", "screaming", "glowing", "ascending", "fading", 
            "resonating", "vibrating", "folding", "blooming", "igniting"
        ]
        
        # Get theme fragment for chapter
        fragments = chapter_themes.get(chapter, ["Eternal Truth", "Divine Presence", "Cosmic Mystery"])
        theme_base = random.choice(fragments)
        modifier = random.choice(modifiers)
        
        # Combine with vibe for unique topic
        # Strategy: Mix the base theme with the modifier OR the vibe, random choice
        if random.random() > 0.5:
             return f"{theme_base} {modifier}"
        else:
             return f"{theme_base} - {vibe} Mode"


    def record_post(self, subliminal: str, visual_style: str, composition: str = "", engagement_score: float = 0.0):
        """
        Records the actual output of a post to update history.
        
        Args:
            subliminal: The subliminal phrase used
            visual_style: The visual style/prompt used
            composition: The camera angle/composition used (NEW)
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
        
        if composition:
            self.state.setdefault("composition_history", []).append(composition)
            if len(self.state["composition_history"]) > 10:
                self.state["composition_history"].pop(0)
                
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
