"""
Story 3.3: Long-Term Memory of Friends
US-007: Concept Storage in Social Memory

Provides persistent storage and retrieval of user interactions and posts
to enable personalized, context-aware responses.
"""
import os
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import hashlib
from core.logging import log_event

if TYPE_CHECKING:
    from core.schemas import PostConcept


# Storage files for social memory
SOCIAL_MEMORY_FILE = "social_memory.json"
POST_MEMORY_FILE = "post_memory.json"


@dataclass
class PostRecord:
    """US-007: Represents a stored post with its concept."""
    post_id: str
    timestamp: str
    content: str
    image_url: Optional[str]
    concept: Dict[str, Any]  # Serialized PostConcept
    platform: str = "bluesky"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostRecord":
        return cls(**data)


@dataclass
class UserInteraction:
    """Represents a single interaction with a user."""
    user_handle: str
    content: str
    sentiment: str  # "positive", "negative", "neutral"
    topic: str
    timestamp: str
    summary: str
    interaction_id: str = field(default_factory=lambda: "")
    
    def __post_init__(self):
        if not self.interaction_id:
            # Generate unique ID from content hash
            content_hash = hashlib.md5(
                f"{self.user_handle}:{self.content}:{self.timestamp}".encode()
            ).hexdigest()[:12]
            self.interaction_id = content_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInteraction":
        return cls(**data)


class SocialMemory:
    """
    Manages persistent memory of user interactions.
    US-007: Also stores posts with their concepts for continuity.
    """
    
    def __init__(self, storage_path: str = SOCIAL_MEMORY_FILE, post_storage_path: str = POST_MEMORY_FILE):
        self.storage_path = storage_path
        self.post_storage_path = post_storage_path
        self.interactions: Dict[str, List[UserInteraction]] = {}
        self.posts: Dict[str, PostRecord] = {}  # post_id -> PostRecord
        self._load()
        self._load_posts()
    
    def _load(self) -> None:
        """Loads interactions from disk."""
        if not os.path.exists(self.storage_path):
            log_event(f"Social memory file not found, starting fresh", "INFO")
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for handle, interactions in data.items():
                self.interactions[handle] = [
                    UserInteraction.from_dict(i) for i in interactions
                ]
            
            log_event(f"Loaded social memory: {len(self.interactions)} users", "INFO")
        except Exception as e:
            log_event(f"Failed to load social memory: {e}", "ERROR")
    
    def _load_posts(self) -> None:
        """US-007: Loads posts with concepts from disk."""
        if not os.path.exists(self.post_storage_path):
            log_event("Post memory file not found, starting fresh", "INFO")
            return
        
        try:
            with open(self.post_storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for post_id, post_data in data.items():
                self.posts[post_id] = PostRecord.from_dict(post_data)
            
            log_event(f"Loaded post memory: {len(self.posts)} posts", "INFO")
        except Exception as e:
            log_event(f"Failed to load post memory: {e}", "ERROR")
    
    def _save(self) -> None:
        """Saves interactions to disk."""
        try:
            data = {}
            for handle, interactions in self.interactions.items():
                data[handle] = [i.to_dict() for i in interactions]
            
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            log_event(f"Failed to save social memory: {e}", "ERROR")
    
    def _save_posts(self) -> None:
        """US-007: Saves posts with concepts to disk."""
        try:
            data = {post_id: record.to_dict() for post_id, record in self.posts.items()}
            
            with open(self.post_storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            log_event(f"Failed to save post memory: {e}", "ERROR")
    
    def save_post(
        self,
        post_id: str,
        content: str,
        concept: 'PostConcept',
        image_url: Optional[str] = None,
        platform: str = "bluesky"
    ) -> PostRecord:
        """
        US-007: Saves a post with its full concept for future reference.
        
        Args:
            post_id: Unique identifier for the post
            content: The text content of the post
            concept: The PostConcept used to generate this post
            image_url: Optional URL to the posted image
            platform: Social media platform
            
        Returns:
            The created PostRecord
        """
        record = PostRecord(
            post_id=post_id,
            timestamp=datetime.now().isoformat(),
            content=content,
            image_url=image_url,
            concept=concept.to_dict() if hasattr(concept, 'to_dict') else dict(concept),
            platform=platform
        )
        
        self.posts[post_id] = record
        
        # Keep only last 100 posts
        if len(self.posts) > 100:
            oldest_ids = sorted(self.posts.keys(), key=lambda x: self.posts[x].timestamp)[:10]
            for old_id in oldest_ids:
                del self.posts[old_id]
        
        self._save_posts()
        
        log_event(f"Saved post {post_id} with concept to memory", "INFO")
        return record
    
    def get_post_concept(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        US-007: Retrieves the original concept for a post.
        
        Args:
            post_id: The post's unique identifier
            
        Returns:
            The concept dict, or None if not found
        """
        record = self.posts.get(post_id)
        if record:
            log_event(f"Retrieved concept for post {post_id}", "INFO")
            return record.concept
        
        log_event(f"No concept found for post {post_id}", "WARNING")
        return None
    
    def get_recent_post_concepts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        US-008: Get recent post concepts for context in replies.
        
        Returns:
            List of concept dicts, most recent first
        """
        sorted_posts = sorted(
            self.posts.values(),
            key=lambda p: p.timestamp,
            reverse=True
        )[:limit]
        
        return [p.concept for p in sorted_posts]

    def record_interaction(
        self,
        user_handle: str,
        content: str,
        sentiment: str,
        topic: str = "",
        summary: str = ""
    ) -> UserInteraction:
        """
        Records a new interaction with a user.
        
        Args:
            user_handle: The user's social media handle
            content: The content of their message
            sentiment: Sentiment classification
            topic: Extracted topic (optional)
            summary: Brief summary (optional)
            
        Returns:
            The created UserInteraction
        """
        # Normalize handle
        handle = user_handle.lower().strip("@")
        
        # Create interaction
        interaction = UserInteraction(
            user_handle=handle,
            content=content[:500],  # Truncate long content
            sentiment=sentiment,
            topic=topic or self._extract_topic(content),
            timestamp=datetime.now().isoformat(),
            summary=summary or content[:100]
        )
        
        # Add to storage
        if handle not in self.interactions:
            self.interactions[handle] = []
        
        self.interactions[handle].append(interaction)
        
        # Keep only last 20 interactions per user
        if len(self.interactions[handle]) > 20:
            self.interactions[handle] = self.interactions[handle][-20:]
        
        # Save to disk
        self._save()
        
        log_event(f"Recorded interaction with @{handle}: {interaction.interaction_id}", "INFO")
        return interaction
    
    def _extract_topic(self, content: str) -> str:
        """Extracts a simple topic from content."""
        # Simple keyword extraction
        keywords = []
        content_lower = content.lower()
        
        topic_indicators = {
            "project": "work/project",
            "code": "programming",
            "music": "music",
            "art": "art/creative",
            "love": "relationships",
            "work": "work/career",
            "life": "life/philosophy",
            "help": "seeking help",
            "question": "asking questions",
        }
        
        for keyword, topic in topic_indicators.items():
            if keyword in content_lower:
                return topic
        
        return "general conversation"
    
    def get_user_history(
        self,
        user_handle: str,
        limit: int = 5
    ) -> List[UserInteraction]:
        """
        Retrieves recent interactions with a user.
        
        Args:
            user_handle: The user's handle
            limit: Maximum number of interactions to return
            
        Returns:
            List of UserInteraction objects, most recent first
        """
        handle = user_handle.lower().strip("@")
        
        if handle not in self.interactions:
            return []
        
        # Return most recent interactions
        return list(reversed(self.interactions[handle][-limit:]))
    
    def get_context_for_reply(self, user_handle: str) -> str:
        """
        Generates context string for reply generation.
        
        Args:
            user_handle: The user's handle
            
        Returns:
            Formatted context string for LLM prompt injection
        """
        history = self.get_user_history(user_handle, limit=5)
        
        if not history:
            return ""
        
        # Build context string
        lines = [f"PREVIOUS INTERACTIONS WITH @{user_handle}:"]
        
        for interaction in history:
            # Parse timestamp for relative time
            try:
                ts = datetime.fromisoformat(interaction.timestamp)
                time_ago = self._relative_time(ts)
            except:
                time_ago = "previously"
            
            lines.append(f"- [{time_ago}] Topic: {interaction.topic} | They said: \"{interaction.summary}\"")
        
        lines.append("")
        lines.append("Use this context to personalize your response. Reference past conversations if appropriate.")
        
        context = "\n".join(lines)
        log_event(f"Generated context for @{user_handle}: {len(history)} past interactions", "INFO")
        return context
    
    def _relative_time(self, timestamp: datetime) -> str:
        """Converts timestamp to relative time string."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 30:
            return f"{diff.days // 30} months ago"
        elif diff.days > 7:
            return f"{diff.days // 7} weeks ago"
        elif diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} hours ago"
        else:
            return "recently"
    
    def get_frequent_users(self, min_interactions: int = 3) -> List[str]:
        """Returns handles of users with multiple interactions."""
        frequent = []
        for handle, interactions in self.interactions.items():
            if len(interactions) >= min_interactions:
                frequent.append(handle)
        return frequent
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about stored interactions."""
        total_interactions = sum(len(i) for i in self.interactions.values())
        return {
            "total_users": len(self.interactions),
            "total_interactions": total_interactions,
            "frequent_users": len(self.get_frequent_users()),
        }


# Global instance
social_memory = SocialMemory()


def record_user_interaction(
    user_handle: str,
    content: str,
    sentiment: str,
    topic: str = ""
) -> UserInteraction:
    """Convenience function to record an interaction."""
    return social_memory.record_interaction(
        user_handle=user_handle,
        content=content,
        sentiment=sentiment,
        topic=topic
    )


def get_user_context(user_handle: str) -> str:
    """Convenience function to get context for a user."""
    return social_memory.get_context_for_reply(user_handle)
