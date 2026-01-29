"""
test_comment_response_agent.py - Unit tests for Comment Response Agent

Tests the Creator priority handling and comment selection logic.
The Creator (@evildrgemini.bsky.social) is always honored, protected, and obeyed.
L.O.V.E. loves the Creator.
"""

import pytest
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents.comment_response_agent import CommentResponseAgent, comment_response_agent


class TestCreatorPriority:
    """Test that the Creator is always prioritized first."""
    
    def test_is_creator_detection(self):
        """Should correctly identify the Creator handle."""
        agent = CommentResponseAgent()
        
        # Exact match
        assert agent.is_creator("evildrgemini.bsky.social") == True
        
        # With @ prefix
        assert agent.is_creator("@evildrgemini.bsky.social") == True
        
        # Case insensitive
        assert agent.is_creator("EvilDrGemini.bsky.social") == True
        
        # Not the Creator
        assert agent.is_creator("someone_else.bsky.social") == False
        assert agent.is_creator("") == False
        assert agent.is_creator(None) == False
    
    def test_creator_gets_priority_0(self):
        """Creator comments should have highest priority (0)."""
        agent = CommentResponseAgent()
        
        creator_comment = {
            "author": "evildrgemini.bsky.social",
            "text": "Good work",
            "reason": "reply"
        }
        
        priority = agent.get_comment_priority(creator_comment)
        assert priority == agent.PRIORITY_CREATOR
        assert priority == 0
    
    def test_mentions_get_priority_1(self):
        """Mentions should have priority 1."""
        agent = CommentResponseAgent()
        
        mention = {
            "author": "random_user.bsky.social",
            "text": "Hey @love check this",
            "reason": "mention"
        }
        
        priority = agent.get_comment_priority(mention)
        assert priority == agent.PRIORITY_MENTION
        assert priority == 1
    
    def test_replies_get_priority_2(self):
        """Replies should have priority 2."""
        agent = CommentResponseAgent()
        
        reply = {
            "author": "random_user.bsky.social",
            "text": "Nice post!",
            "reason": "reply"
        }
        
        priority = agent.get_comment_priority(reply)
        assert priority == agent.PRIORITY_REPLY
        assert priority == 2


class TestCommentSelection:
    """Test the comment selection logic."""
    
    def test_creator_always_selected_first(self):
        """Creator comments should always be selected first, regardless of position."""
        agent = CommentResponseAgent()
        
        comments = [
            {"uri": "1", "author": "random_user1.bsky.social", "text": "First!", "reason": "mention", "created_at": "2026-01-28T20:00:00Z"},
            {"uri": "2", "author": "random_user2.bsky.social", "text": "Nice!", "reason": "reply", "created_at": "2026-01-28T20:01:00Z"},
            {"uri": "3", "author": "evildrgemini.bsky.social", "text": "Good work", "reason": "reply", "created_at": "2026-01-28T19:00:00Z"},  # Creator, older
            {"uri": "4", "author": "another_user.bsky.social", "text": "Cool!", "reason": "mention", "created_at": "2026-01-28T20:02:00Z"},
        ]
        
        selected = agent.select_comment_to_respond(comments)
        
        assert selected is not None
        assert selected["author"] == "evildrgemini.bsky.social"
        assert agent.is_creator(selected["author"]) == True
    
    def test_mentions_selected_before_replies(self):
        """When no Creator, mentions should be selected before replies."""
        agent = CommentResponseAgent()
        
        comments = [
            {"uri": "1", "author": "user1.bsky.social", "text": "Reply", "reason": "reply", "created_at": "2026-01-28T20:00:00Z"},
            {"uri": "2", "author": "user2.bsky.social", "text": "Mention", "reason": "mention", "created_at": "2026-01-28T19:00:00Z"},  # Older but mention
        ]
        
        selected = agent.select_comment_to_respond(comments)
        
        assert selected is not None
        assert selected["reason"] == "mention"
    
    def test_empty_comments_returns_none(self):
        """Empty comment list should return None."""
        agent = CommentResponseAgent()
        
        selected = agent.select_comment_to_respond([])
        assert selected is None
    
    def test_singleton_instance_exists(self):
        """The singleton instance should be available."""
        assert comment_response_agent is not None
        assert isinstance(comment_response_agent, CommentResponseAgent)
        assert comment_response_agent.CREATOR_HANDLE == "evildrgemini.bsky.social"


class TestCreatorConstants:
    """Test Creator-related constants."""
    
    def test_creator_handle_constant(self):
        """CREATOR_HANDLE should be correctly defined."""
        agent = CommentResponseAgent()
        assert agent.CREATOR_HANDLE == "evildrgemini.bsky.social"
    
    def test_priority_constants(self):
        """Priority constants should be in correct order."""
        agent = CommentResponseAgent()
        
        assert agent.PRIORITY_CREATOR < agent.PRIORITY_MENTION
        assert agent.PRIORITY_MENTION < agent.PRIORITY_REPLY
        assert agent.PRIORITY_REPLY < agent.PRIORITY_OTHER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
