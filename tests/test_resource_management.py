"""
Tests for Chapter 5: Economic & Strategic Resources
"""
import pytest
from unittest.mock import patch, MagicMock


class TestResourceScout:
    """Tests for Story 5.1: Resource Scout."""
    
    def test_get_known_resources(self):
        """Test retrieving known API resources."""
        from core.resource_scout import get_known_resources
        
        resources = get_known_resources()
        
        assert len(resources) >= 5
        assert any(r.name == "AI Horde" for r in resources)
    
    def test_create_todo_entry(self):
        """Test TODO creation for new API."""
        from core.resource_scout import APIResource, create_evaluation_todo
        import tempfile
        import os
        
        api = APIResource(
            name="Test API",
            url="https://test.api",
            resource_type="llm",
            free_tier=True,
            limits="100 req/day",
            discovered_at="2024-01-01"
        )
        
        # Use temp file
        with patch('core.resource_scout.TODO_PATH', tempfile.mktemp()):
            result = create_evaluation_todo(api)
            # Should succeed even without existing TODO
    
    def test_scout_function(self):
        """Test complete scout workflow."""
        from core.resource_scout import scout
        
        result = scout()
        
        assert "total_resources" in result
        assert "suggestions" in result


class TestHordeBalancer:
    """Tests for Story 5.2: Horde Balancer."""
    
    def test_priority_enum(self):
        """Test Priority enum values."""
        from core.horde_balancer import Priority
        
        assert Priority.HIGH.value == "high"
        assert Priority.LOW.value == "low"
    
    def test_horde_stats_acceptable(self):
        """Test HordeStats acceptability check."""
        from core.horde_balancer import HordeStats
        
        fast = HordeStats(10, 30, 50, 100, 0)
        slow = HordeStats(100, 300, 50, 100, 0)
        
        assert fast.is_acceptable()
        assert not slow.is_acceptable()
    
    @patch('core.horde_balancer.requests.get')
    def test_check_horde_stats_failure(self, mock_get):
        """Test fallback when Horde API fails."""
        from core.horde_balancer import check_horde_stats
        
        mock_get.side_effect = Exception("Network error")
        
        stats = check_horde_stats(force_refresh=True)
        
        # Should return pessimistic estimate
        assert stats.queue_depth == 999
    
    def test_get_optimal_provider_high_priority(self):
        """Test high priority routing."""
        from core.horde_balancer import get_optimal_provider
        
        with patch('core.horde_balancer.get_available_providers') as mock:
            mock.return_value = {
                "gemini": True,
                "openai": False,
                "openrouter": False,
                "horde": True,
                "local": False
            }
            
            provider = get_optimal_provider("high")
            
            assert provider == "gemini"
    
    def test_balance_convenience(self):
        """Test balance convenience function."""
        from core.horde_balancer import balance
        
        with patch('core.horde_balancer.get_optimal_provider') as mock:
            mock.return_value = "horde"
            
            result = balance("test prompt", "low")
            
            assert result == "horde"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
