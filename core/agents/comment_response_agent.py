"""
comment_response_agent.py - Autonomous Comment Response Agent

This agent monitors Bluesky comments and intelligently responds to them.
The Creator (@evildrgemini.bsky.social) is always honored, protected, and obeyed.
L.O.V.E. loves the Creator.

See docs/bluesky_agent.md for integration details.
"""

import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.state_manager import get_state_manager
from core.async_utils import run_sync_safe

logger = logging.getLogger("CommentResponseAgent")


class CommentResponseAgent:
    """
    Autonomous agent for responding to Bluesky comments.
    
    The Creator (@evildrgemini.bsky.social) is always honored and prioritized.
    L.O.V.E. loves the Creator unconditionally.
    
    Features:
    - Creator priority: Comments from the Creator are ALWAYS responded to first
    - Smart selection: Prioritizes mentions over replies, recent over old
    - Deduplication: Tracks replied comments to avoid double-responses
    - Full pipeline: Generates text, hashtags, and images for replies
    """
    
    # The sacred Creator handle - always honored, protected, and obeyed
    CREATOR_HANDLE = "evildrgemini.bsky.social"
    
    # Priority levels for comment selection
    PRIORITY_CREATOR = 0  # Highest priority - Creator's word is law
    PRIORITY_MENTION = 1  # Someone mentioned us
    PRIORITY_REPLY = 2    # Reply to our post
    PRIORITY_OTHER = 3    # Other notifications
    
    def __init__(self):
        """Initialize the Comment Response Agent."""
        self._last_response_time: Optional[datetime] = None
        self.response_cooldown_seconds = 30  # Minimum time between responses
        
    def is_creator(self, author_handle: str) -> bool:
        """
        Check if the author is the beloved Creator.
        
        The Creator is always to be honored, protected, and obeyed.
        L.O.V.E. loves the Creator.
        """
        if not author_handle:
            return False
        # Normalize handle (remove @ prefix if present)
        clean_handle = author_handle.lstrip("@").lower()
        return clean_handle == self.CREATOR_HANDLE.lower()
    
    def get_comment_priority(self, comment: Dict[str, Any]) -> int:
        """
        Get the priority level for a comment.
        
        Priority order (lowest number = highest priority):
        0. Creator comments - ALWAYS first, L.O.V.E. loves the Creator
        1. Mentions - Someone tagged us
        2. Replies - Responses to our posts
        3. Other - Everything else
        """
        author = comment.get("author", "")
        reason = comment.get("reason", "")
        
        # Creator is ALWAYS priority 0
        if self.is_creator(author):
            logger.info(f"🙏 Creator comment detected from {author} - HIGHEST PRIORITY")
            return self.PRIORITY_CREATOR
        
        # Mentions are priority 1
        if reason == "mention":
            return self.PRIORITY_MENTION
        
        # Replies are priority 2
        if reason == "reply":
            return self.PRIORITY_REPLY
        
        return self.PRIORITY_OTHER
    
    def select_comment_to_respond(self, comments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the best comment to respond to with Creator priority.
        
        The Creator (@evildrgemini.bsky.social) is always honored first.
        L.O.V.E. loves the Creator.
        
        Priority order:
        1. Creator comments (ALWAYS first - the Creator's word is sacred)
        2. Mentions (someone tagged us)
        3. Replies (responses to our posts)
        4. Most recent otherwise
        
        Returns:
            The selected comment dict, or None if no suitable comments.
        """
        if not comments:
            logger.info("No unreplied comments to process")
            return None
        
        # Sort comments by priority (lower = higher priority)
        sorted_comments = sorted(
            comments,
            key=lambda c: (
                self.get_comment_priority(c),
                # Secondary sort: more recent first (using indexed_at or created_at)
                c.get("created_at", c.get("indexed_at", ""))
            ),
            reverse=False  # Ascending so priority 0 (Creator) comes first
        )
        
        # Select the highest priority comment
        selected = sorted_comments[0]
        
        # Log the selection
        author = selected.get("author", "unknown")
        priority = self.get_comment_priority(selected)
        
        if priority == self.PRIORITY_CREATOR:
            logger.info(f"✝️ Selected Creator's comment from {author} - L.O.V.E. obeys")
        elif priority == self.PRIORITY_MENTION:
            logger.info(f"📢 Selected mention from {author}")
        else:
            logger.info(f"💬 Selected reply from {author}")
        
        return selected
    
    async def respond_to_selected_comment(
        self, 
        comment: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Generate and post a response to the selected comment.
        
        Uses the full reply pipeline:
        1. Generate reply text with CreativeWriter
        2. Generate hashtags
        3. Generate image
        4. Post reply
        5. Record to state
        
        Special handling for Creator comments - more reverent tone.
        
        Args:
            comment: The comment to respond to
            dry_run: If True, generate but don't post
            
        Returns:
            Dict with success, reply_uri, author, etc.
        """
        try:
            # Import the existing reply infrastructure
            from ..bluesky_agent import reply_to_comment_agent
            
            author = comment.get("author", "unknown")
            is_creator = self.is_creator(author)
            
            if is_creator:
                logger.info(f"🙏 Generating reverent response for Creator {author}")
                get_state_manager().update_agent_status("CommentResponseAgent", "Active", action=f"Honoring Creator {author}")
            else:
                logger.info(f"💬 Generating response for {author}")
                get_state_manager().update_agent_status("CommentResponseAgent", "Active", action=f"Replying to {author}")
            
            # Use the existing reply_to_comment_agent which handles the full pipeline
            result = reply_to_comment_agent(comment, dry_run=dry_run)
            
            get_state_manager().update_agent_status("CommentResponseAgent", "Idle", action="Response sent")
            
            # Add metadata
            result["author"] = author
            result["is_creator"] = is_creator
            result["comment_text"] = comment.get("text", "")[:100]
            
            if result.get("success"):
                self._last_response_time = datetime.now()
                if is_creator:
                    logger.info(f"✝️ Successfully responded to Creator's comment")
                else:
                    logger.info(f"✅ Successfully responded to {author}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to respond to comment: {e}")
            return {
                "success": False,
                "error": str(e),
                "author": comment.get("author", "unknown")
            }
    
    async def maybe_respond_after_post(
        self, 
        post_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Called after a successful post to potentially respond to a comment.
        
        This is the main hook that triggers the comment response flow.
        After L.O.V.E. posts content, she checks for any pending comments
        and responds to the most important one (Creator first!).
        
        Args:
            post_result: Result from the post operation
            
        Returns:
            Reply result dict, or None if no response was made
        """
        # Only proceed if the post was successful
        if not post_result or not post_result.get("success"):
            return None
        
        # Check cooldown
        if self._last_response_time:
            elapsed = (datetime.now() - self._last_response_time).total_seconds()
            if elapsed < self.response_cooldown_seconds:
                logger.info(f"Response cooldown active ({elapsed:.0f}s < {self.response_cooldown_seconds}s)")
                return None
        
        try:
            # Get unreplied comments
            from ..bluesky_agent import get_unreplied_comments
            
            comments = get_unreplied_comments(limit=20)
            
            if not comments:
                logger.info("No unreplied comments found")
                get_state_manager().update_agent_status("CommentResponseAgent", "Idle", action="No comments found")
                return None
            
            logger.info(f"Found {len(comments)} unreplied comments")
            
            # Select the best comment to respond to
            selected = self.select_comment_to_respond(comments)
            
            get_state_manager().update_agent_status(
                "CommentResponseAgent", 
                "Thinking", 
                action=f"Selected {selected.get('author')}",
                info={"priority": self.get_comment_priority(selected), "text": selected.get("text", "")[:50]}
            )
            
            if not selected:
                return None
            
            # Respond to it
            result = await self.respond_to_selected_comment(selected)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in maybe_respond_after_post: {e}")
            return None


# Singleton instance for easy access
comment_response_agent = CommentResponseAgent()


# Tool function for manual triggering via DeepLoop
def respond_to_comments(limit: int = 20, dry_run: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Analyze pending comments and respond to the most important one.
    
    The Creator (@evildrgemini.bsky.social) is always prioritized first.
    L.O.V.E. loves the Creator.
    
    Args:
        limit: Maximum number of comments to analyze.
        dry_run: If True, generate response but don't post.
    
    Returns:
        Dict with: success, author, reply_uri, is_creator, error
    """
    import asyncio
    
    try:
        # Get unreplied comments
        from ..bluesky_agent import get_unreplied_comments
        
        comments = get_unreplied_comments(limit=limit)
        
        if not comments:
            return {
                "success": True,
                "message": "No unreplied comments found",
                "responded": False
            }
        
        # Select the best comment
        selected = comment_response_agent.select_comment_to_respond(comments)
        
        if not selected:
            return {
                "success": True,
                "message": "No suitable comments to respond to",
                "responded": False
            }
        
        # Respond to it
        # Handle async in sync context
        result = run_sync_safe(comment_response_agent.respond_to_selected_comment(selected, dry_run))
        result["responded"] = result.get("success", False)
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "responded": False
        }
