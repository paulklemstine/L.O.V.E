"""
test_bluesky_agent.py - Tests for Bluesky Social Media Agent

Tests the Bluesky integration including:
- Posting to Bluesky
- Rate limiting / cooldown
- Timeline fetching
- Content generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import bluesky_agent


class TestPostCooldown:
    """Test post rate limiting."""
    
    def test_cooldown_enforced(self):
        """Test that cooldown prevents rapid posting."""
        # Simulate recent post
        bluesky_agent._last_post_time = datetime.now()
        
        result = bluesky_agent.post_to_bluesky("Test post")
        
        assert result["success"] == False
        assert "cooldown" in result["error"].lower()
    
    def test_cooldown_expired(self):
        """Test that cooldown allows posting after expiry."""
        # Simulate old post (well past cooldown)
        bluesky_agent._last_post_time = datetime.now() - timedelta(hours=1)
        
        # This will fail for other reasons (no client), but not cooldown
        result = bluesky_agent.post_to_bluesky("Test post")
        
        # If it failed, should not be cooldown related
        if not result["success"]:
            assert "cooldown" not in (result.get("error") or "").lower()


class TestPostValidation:
    """Test post content validation."""
    
    def test_text_length_validation(self):
        """Test that overly long text is rejected."""
        # Reset cooldown
        bluesky_agent._last_post_time = None
        
        long_text = "x" * 301
        
        result = bluesky_agent.post_to_bluesky(long_text)
        
        assert result["success"] == False
        assert "300" in result["error"]


class TestBlueskyTimeline:
    """Test timeline fetching."""
    
    def test_limit_clamping(self):
        """Test that limit is clamped to valid range."""
        with patch.object(bluesky_agent, '_get_bluesky_client') as mock_client:
            mock = Mock()
            mock.get_timeline.return_value = Mock(feed=[])
            mock_client.return_value = mock
            
            bluesky_agent.get_bluesky_timeline(limit=100)  # Over limit
            
            # Should be called with clamped limit
            mock.get_timeline.assert_called_once_with(limit=50)


class TestBlueskySearch:
    """Test Bluesky search."""
    
    def test_search_with_query(self):
        """Test search returns results."""
        with patch.object(bluesky_agent, '_get_bluesky_client') as mock_client:
            # Setup mock
            mock = Mock()
            mock.app.bsky.feed.search_posts.return_value = Mock(posts=[])
            mock_client.return_value = mock
            
            result = bluesky_agent.search_bluesky("test query")
            
            assert result["success"] == True
            assert isinstance(result["posts"], list)


class TestContentGeneration:
    """Test AI-powered content generation."""
    
    def test_generate_post_content(self):
        """Test generating post content with LLM."""
        with patch('core.bluesky_agent.get_llm_client') as mock_llm, \
             patch('core.bluesky_agent.get_persona_extractor') as mock_persona:
            
            # Mock LLM response
            llm = Mock()
            llm.generate_json.return_value = {
                "text": "Catch the wave! 🌊",
                "hashtags": ["beach", "vibes"]
            }
            mock_llm.return_value = llm
            
            # Mock persona
            persona = Mock()
            persona.get_persona_context.return_value = "Beach goddess persona"
            mock_persona.return_value = persona
            
            result = bluesky_agent.generate_post_content(topic="beach vibes")
            
            assert result["success"] == True
            assert result["text"] is not None
            assert isinstance(result["hashtags"], list)
