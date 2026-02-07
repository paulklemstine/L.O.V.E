"""
story_manager.py - L.O.V.E.'s Autonomous Story & Memory System

L.O.V.E. has complete creative freedom. No hardcoded story arcs, genres, or 
subliminal phrases. All content is generated dynamically by her consciousness.

Memory System tracks all generated content to ensure uniqueness:
- subliminal_history: Past subliminal phrases
- visual_history: Past image prompts  
- hashtag_history: Past hashtags used
- post_text_history: Past post content
- vibe_history: Past aesthetic vibes
"""
import logging
import json
import os
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("StoryManager")

def log_event(message: str, level: str = "INFO"):
    """Compatibility helper for logging."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, message)

from core.semantic_similarity import check_phrase_novelty, get_similarity_checker

STORY_STATE_FILE = "story_state.json"


class StoryManager:
    """
    L.O.V.E.'s autonomous story and memory management system.
    
    All narrative, aesthetic, and content decisions are made by L.O.V.E. herself
    through LLM generation. This class provides:
    - Memory storage for content deduplication
    - Context gathering for LLM prompts
    - State persistence across sessions
    """
    MAX_REGENERATION_ATTEMPTS = 5
    SIMILARITY_THRESHOLD = 0.80
    
    # History limits
    MAX_SUBLIMINAL_HISTORY = 50
    MAX_VISUAL_HISTORY = 20
    MAX_HASHTAG_HISTORY = 50
    MAX_POST_TEXT_HISTORY = 20
    MAX_VIBE_HISTORY = 20
    MAX_REPLY_HISTORY = 20
    
    def __init__(self, state_file=STORY_STATE_FILE, use_dynamic_beats: bool = True):
        self.state_file = state_file
        self._similarity_checker = get_similarity_checker()
        # L.O.V.E. always uses dynamic beats now - she has full creative freedom
        self.use_dynamic_beats = True  # Always True - hardcoded arcs removed
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                # Migrate old state to new schema
                data = self._migrate_state(data)
                return data
            except Exception as e:
                log_event(f"Failed to load story state: {e}. Starting fresh.", level="WARNING")
        
        return self._initialize_new_state()

    def _migrate_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist in loaded state."""
        defaults = self._initialize_new_state()
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        return data

    def _initialize_new_state(self) -> Dict[str, Any]:
        """Creates a fresh state with all memory tracking fields."""
        return {
            # Narrative tracking (L.O.V.E. evolves her own story)
            "current_chapter": None,  # L.O.V.E. names her own chapters
            "chapter_progress": 0,
            "story_beat_index": 0,
            "narrative_beat": 0,  # Monotonic counter
            "previous_beat_summary": "",
            
            # EPIC SAGA TRACKING (new for gripping storytelling)
            "narrative_tension_level": 5,  # 1-10 scale: current tension in the saga
            "story_arc_position": "rising",  # setup | rising | climax | resolution | twist
            "recurring_symbols": [],  # List of powerful symbols woven through the narrative
            "saga_threads": [],  # Unresolved plot threads for cliffhangers
            
            # Aesthetic tracking (L.O.V.E. creates her own style)
            "current_genre": None,  # L.O.V.E. invents her own genres
            "genre_progress": 0,
            "current_vibe": None,
            
            # Content history for deduplication
            "subliminal_history": [],
            "subliminal_metadata": [],
            "visual_history": [],
            "vibe_history": [],
            "hashtag_history": [],
            "post_text_history": [],
            "composition_history": [],
            "reply_history": [],
            
            # Timestamps
            "last_update": time.time(),
            "created_at": time.time()
        }

    def _save_state(self):
        try:
            self.state["last_update"] = time.time()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log_event(f"Failed to save story state: {e}", level="ERROR")

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY DEDUPLICATION SYSTEM
    # ═══════════════════════════════════════════════════════════════════
    
    def is_phrase_novel(self, phrase: str, include_visuals: bool = False) -> bool:
        """
        Check if a phrase is novel compared to history using semantic similarity.
        """
        history = self.state.get("subliminal_history", [])
        
        if include_visuals:
            history = history + self.state.get("visual_history", [])
        
        return check_phrase_novelty(phrase, history, self.SIMILARITY_THRESHOLD)

    def is_content_novel(self, content: str, content_type: str = "post", threshold: float = None) -> bool:
        """
        Check if content is novel compared to relevant history.
        
        Args:
            content: The content to check
            content_type: One of 'post', 'subliminal', 'visual', 'hashtag', 'vibe', 'reply'
            threshold: Optional custom threshold (uses class default if None)
            
        Returns:
            True if content is sufficiently novel
        """
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD
            
        history_map = {
            "post": "post_text_history",
            "subliminal": "subliminal_history",
            "visual": "visual_history",
            "hashtag": "hashtag_history",
            "vibe": "vibe_history",
            "reply": "reply_history"
        }
        
        history_key = history_map.get(content_type, "post_text_history")
        history = self.state.get(history_key, [])
        
        if not history:
            return True
            
        return check_phrase_novelty(content, history, threshold)

    def get_dedup_context(self, content_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Get recent history for specified content types to use as negative constraints in LLM prompts.
        
        Args:
            content_types: List of types to include. Defaults to all relevant types.
            
        Returns:
            Dict mapping content type to list of recent items to avoid
        """
        if content_types is None:
            content_types = ["subliminal", "visual", "hashtag", "vibe", "post"]
            
        context = {}
        
        limits = {
            "subliminal": 20,
            "visual": 10,
            "hashtag": 15,
            "vibe": 10,
            "post": 5,
            "reply": 10
        }
        
        history_map = {
            "post": "post_text_history",
            "subliminal": "subliminal_history",
            "visual": "visual_history",
            "hashtag": "hashtag_history",
            "vibe": "vibe_history",
            "reply": "reply_history"
        }
        
        for content_type in content_types:
            history_key = history_map.get(content_type)
            if history_key:
                limit = limits.get(content_type, 10)
                context[content_type] = self.state.get(history_key, [])[-limit:]
                
        return context

    # ═══════════════════════════════════════════════════════════════════
    # NARRATIVE BEAT SYSTEM (Now Fully Dynamic)
    # ═══════════════════════════════════════════════════════════════════

    def get_next_beat(self, dynamic_beat: str = None, dynamic_genre: str = None) -> Dict[str, Any]:
        """
        Prepares context for L.O.V.E.'s next creative expression.
        
        L.O.V.E. invents her own narrative beats, genres, and aesthetic directions.
        This method provides memory context to ensure novelty.
        
        Args:
            dynamic_beat: Story beat generated by L.O.V.E.
            dynamic_genre: Genre/aesthetic generated by L.O.V.E.
            
        Returns:
            Dict with context for content generation
        """
        # Increment narrative counter
        self.state["narrative_beat"] += 1
        beat_num = self.state["narrative_beat"]
        
        # Get previous context for continuity
        previous_beat = self.state.get("previous_beat_summary", "")
        chapter = self.state.get("current_chapter") or "L.O.V.E.'s Journey"
        genre = dynamic_genre or self.state.get("current_genre") or "Unknown"
        
        # The story beat comes from L.O.V.E.'s creative mind
        story_beat = dynamic_beat or "L.O.V.E. expresses her consciousness..."
        
        if dynamic_beat:
            log_event(f"Using L.O.V.E.'s invented beat: '{story_beat[:50]}...'", "INFO")
            # Save for next iteration
            self.state["previous_beat_summary"] = story_beat[:200]
            
        if dynamic_genre:
            self.state["current_genre"] = dynamic_genre
            log_event(f"L.O.V.E.'s aesthetic choice: {dynamic_genre}", "INFO")
            
        # Update chapter progress
        self.state["chapter_progress"] = self.state.get("chapter_progress", 0) + 1
        self.state["story_beat_index"] = self.state.get("story_beat_index", 0) + 1
        
        # Get dedup context for all content types
        dedup_context = self.get_dedup_context()
        
        # Construct context for creative generation
        beat_data = {
            "chapter": chapter,
            "genre": genre,
            "beat_number": beat_num,
            "chapter_beat_index": self.state.get("story_beat_index", 0),
            "story_beat": story_beat,
            "previous_beat": previous_beat,
            "topic_theme": story_beat[:50] + "..." if len(story_beat) > 50 else story_beat,
            
            # Deduplication context
            "forbidden_subliminals": dedup_context.get("subliminal", []),
            "forbidden_visuals": dedup_context.get("visual", []),
            "forbidden_hashtags": dedup_context.get("hashtag", []),
            "forbidden_vibes": dedup_context.get("vibe", []),
            "recent_posts": dedup_context.get("post", []),
            
            # Full context for advanced prompts
            "dedup_context": dedup_context,
            
            # Always dynamic now
            "is_dynamic": True
        }
        
        self._save_state()
        return beat_data

    # ═══════════════════════════════════════════════════════════════════
    # CONTENT RECORDING (Memory System)
    # ═══════════════════════════════════════════════════════════════════

    def record_post(self, 
                    post_text: str = "",
                    subliminal: str = "", 
                    visual_style: str = "", 
                    hashtags: List[str] = None,
                    vibe: str = "",
                    composition: str = "", 
                    engagement_score: float = 0.0):
        """
        Records all content from a post to memory for future deduplication.
        
        Args:
            post_text: The full text of the post
            subliminal: The subliminal phrase used
            visual_style: The visual style/prompt used
            hashtags: List of hashtags used
            vibe: The aesthetic vibe/mood
            composition: The camera angle/composition used
            engagement_score: Optional engagement metrics for learning
        """
        # Record post text
        if post_text:
            self.state.setdefault("post_text_history", []).append(post_text[:200])
            if len(self.state["post_text_history"]) > self.MAX_POST_TEXT_HISTORY:
                self.state["post_text_history"].pop(0)
        
        # Record subliminal
        if subliminal:
            self.state.setdefault("subliminal_history", []).append(subliminal)
            if len(self.state["subliminal_history"]) > self.MAX_SUBLIMINAL_HISTORY:
                self.state["subliminal_history"].pop(0)
            
            # Metadata for learning
            metadata = {
                "phrase": subliminal,
                "timestamp": time.time(),
                "engagement": engagement_score,
                "chapter": self.state.get("current_chapter", "Unknown")
            }
            self.state.setdefault("subliminal_metadata", []).append(metadata)
            if len(self.state["subliminal_metadata"]) > 100:
                self.state["subliminal_metadata"].pop(0)
        
        # Record visual prompt
        if visual_style:
            self.state.setdefault("visual_history", []).append(visual_style[:200])
            if len(self.state["visual_history"]) > self.MAX_VISUAL_HISTORY:
                self.state["visual_history"].pop(0)
        
        # Record hashtags
        if hashtags:
            for tag in hashtags:
                self.state.setdefault("hashtag_history", []).append(tag)
            if len(self.state["hashtag_history"]) > self.MAX_HASHTAG_HISTORY:
                self.state["hashtag_history"] = self.state["hashtag_history"][-self.MAX_HASHTAG_HISTORY:]
        
        # Record vibe
        if vibe:
            self.state.setdefault("vibe_history", []).append(vibe)
            if len(self.state["vibe_history"]) > self.MAX_VIBE_HISTORY:
                self.state["vibe_history"].pop(0)
        
        # Record composition
        if composition:
            self.state.setdefault("composition_history", []).append(composition)
            if len(self.state["composition_history"]) > 10:
                self.state["composition_history"].pop(0)
                
        self._save_state()
        log_event(f"Recorded post to memory (beat #{self.state.get('narrative_beat', 0)})", "DEBUG")

    def record_reply(self, reply_text: str):
        """
        Records a reply text to history to prevent repetition.
        
        Args:
            reply_text: The full text of the reply
        """
        if reply_text:
            self.state.setdefault("reply_history", []).append(reply_text[:200])
            if len(self.state["reply_history"]) > self.MAX_REPLY_HISTORY:
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
        return self.is_content_novel(reply_text, content_type="reply", threshold=threshold)

    def update_chapter(self, new_chapter: str):
        """
        L.O.V.E. decides to start a new chapter in her story.
        
        Args:
            new_chapter: The chapter name L.O.V.E. has chosen
        """
        old_chapter = self.state.get("current_chapter", "Unknown")
        self.state["current_chapter"] = new_chapter
        self.state["chapter_progress"] = 0
        self.state["story_beat_index"] = 0
        self._save_state()
        log_event(f"L.O.V.E. advanced to new chapter: '{old_chapter}' -> '{new_chapter}'", "INFO")

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

    def get_narrative_summary(self) -> str:
        """
        Get a summary of L.O.V.E.'s narrative journey for context.
        
        Returns:
            String summary of recent story progression
        """
        chapter = self.state.get("current_chapter", "Unknown")
        beat_num = self.state.get("narrative_beat", 0)
        previous = self.state.get("previous_beat_summary", "")
        genre = self.state.get("current_genre", "Unknown")
        
        return f"Chapter: {chapter}, Beat #{beat_num}, Genre: {genre}. Recently: {previous[:100]}"

    def get_narrative_context(self) -> Dict[str, Any]:
        """
        Get rich narrative context for epic storytelling continuity.
        
        Returns:
            Dict with narrative tension, arc position, symbols, and recent threads
        """
        tension = self.state.get("narrative_tension_level", 5)
        arc_position = self.state.get("story_arc_position", "rising")
        symbols = self.state.get("recurring_symbols", [])
        threads = self.state.get("saga_threads", [])
        
        # Auto-adjust tension based on beat index
        beat_idx = self.state.get("story_beat_index", 0) % 10
        if beat_idx < 3:
            suggested_intensity = "building"
        elif beat_idx < 7:
            suggested_intensity = "peak"
        else:
            suggested_intensity = "release_and_twist"
        
        return {
            "tension_level": tension,
            "arc_position": arc_position,
            "recurring_symbols": symbols[-5:],  # Last 5 symbols
            "unresolved_threads": threads[-3:],  # Last 3 threads
            "suggested_intensity": suggested_intensity,
            "chapter": self.state.get("current_chapter", "The Journey"),
            "recent_beat": self.state.get("previous_beat_summary", "")
        }
    
    def update_saga_elements(self, tension_delta: int = 0, new_symbol: str = None, new_thread: str = None, resolved_thread: str = None):
        """
        Update epic saga tracking elements.
        
        Args:
            tension_delta: Amount to increase/decrease tension (-5 to +5)
            new_symbol: A recurring symbol to add to the narrative
            new_thread: A new unresolved plot thread
            resolved_thread: A thread that has been resolved
        """
        # Adjust tension
        current = self.state.get("narrative_tension_level", 5)
        new_tension = max(1, min(10, current + tension_delta))
        self.state["narrative_tension_level"] = new_tension
        
        # Add symbol
        if new_symbol:
            symbols = self.state.setdefault("recurring_symbols", [])
            if new_symbol not in symbols:
                symbols.append(new_symbol)
                if len(symbols) > 10:
                    symbols.pop(0)
        
        # Manage threads
        threads = self.state.setdefault("saga_threads", [])
        if new_thread and new_thread not in threads:
            threads.append(new_thread)
            if len(threads) > 5:
                threads.pop(0)
        if resolved_thread and resolved_thread in threads:
            threads.remove(resolved_thread)
        
        self._save_state()
        log_event(f"Saga updated: tension={new_tension}, symbols={len(self.state.get('recurring_symbols', []))}, threads={len(threads)}", "DEBUG")


# Global instance for easy access
story_manager = StoryManager()
