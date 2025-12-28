"""
Tests for Story 3.3: Long-Term Memory of Friends
"""
import pytest
import tempfile
import os
import json


class TestUserInteraction:
    """Tests for UserInteraction dataclass."""
    
    def test_create_interaction(self):
        """Test creating a user interaction."""
        from core.social_memory import UserInteraction
        
        interaction = UserInteraction(
            user_handle="testuser",
            content="Hello there!",
            sentiment="positive",
            topic="greeting",
            timestamp="2024-01-01T12:00:00",
            summary="Hello there!"
        )
        
        assert interaction.user_handle == "testuser"
        assert interaction.interaction_id != ""  # Auto-generated
    
    def test_to_dict(self):
        """Test serialization to dict."""
        from core.social_memory import UserInteraction
        
        interaction = UserInteraction(
            user_handle="testuser",
            content="Test content",
            sentiment="neutral",
            topic="test",
            timestamp="2024-01-01T12:00:00",
            summary="Test"
        )
        
        d = interaction.to_dict()
        assert d["user_handle"] == "testuser"
        assert "interaction_id" in d


class TestSocialMemory:
    """Tests for SocialMemory class."""
    
    def test_record_and_retrieve(self):
        """Test recording and retrieving interactions."""
        from core.social_memory import SocialMemory
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            memory = SocialMemory(storage_path=temp_path)
            
            # Record interaction
            memory.record_interaction(
                user_handle="@testfriend",
                content="I love your work!",
                sentiment="positive",
                topic="appreciation"
            )
            
            # Retrieve history
            history = memory.get_user_history("testfriend")
            
            assert len(history) == 1
            assert history[0].sentiment == "positive"
        finally:
            os.unlink(temp_path)
    
    def test_handle_normalization(self):
        """Test that handles are normalized."""
        from core.social_memory import SocialMemory
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            memory = SocialMemory(storage_path=temp_path)
            
            # Record with @ prefix
            memory.record_interaction("@TestUser", "Hello", "neutral")
            
            # Retrieve without @ and different case
            history = memory.get_user_history("testuser")
            
            assert len(history) == 1
        finally:
            os.unlink(temp_path)
    
    def test_get_context_for_reply(self):
        """Test context generation for replies."""
        from core.social_memory import SocialMemory
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            memory = SocialMemory(storage_path=temp_path)
            
            # Record some interactions
            memory.record_interaction("@friend", "Working on my project", "positive", "work")
            memory.record_interaction("@friend", "Art is amazing", "positive", "art")
            
            # Get context
            context = memory.get_context_for_reply("friend")
            
            assert "PREVIOUS INTERACTIONS" in context
            assert "friend" in context.lower()
        finally:
            os.unlink(temp_path)
    
    def test_persistence(self):
        """Test that data persists across instances."""
        from core.social_memory import SocialMemory
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and record
            memory1 = SocialMemory(storage_path=temp_path)
            memory1.record_interaction("@user1", "Test message", "neutral")
            
            # New instance should load existing data
            memory2 = SocialMemory(storage_path=temp_path)
            history = memory2.get_user_history("user1")
            
            assert len(history) == 1
        finally:
            os.unlink(temp_path)
    
    def test_limit_history_size(self):
        """Test that history is limited to 20 entries."""
        from core.social_memory import SocialMemory
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            memory = SocialMemory(storage_path=temp_path)
            
            # Add 25 interactions
            for i in range(25):
                memory.record_interaction("@user", f"Message {i}", "neutral")
            
            # Should only keep 20
            assert len(memory.interactions["user"]) == 20
        finally:
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_record_user_interaction(self):
        """Test the convenience function."""
        from core.social_memory import record_user_interaction, social_memory
        
        # This uses the global instance, so we just verify it doesn't error
        interaction = record_user_interaction(
            user_handle="test_conv",
            content="Test content",
            sentiment="positive"
        )
        
        assert interaction is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
