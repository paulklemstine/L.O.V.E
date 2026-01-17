"""
Tests for the security module and code injection functionality.
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestSecurityModule:
    """Tests for core/security.py"""
    
    def test_is_creator_instance_without_key(self):
        """Test that is_creator_instance returns False when key doesn't exist."""
        from core.security import is_creator_instance, clear_creator_cache
        
        clear_creator_cache()
        
        # Point to a non-existent file
        result = is_creator_instance("/nonexistent/path/key.pem")
        assert result is False
    
    def test_is_creator_instance_with_key(self):
        """Test that is_creator_instance returns True when key exists."""
        from core.security import is_creator_instance, clear_creator_cache
        
        clear_creator_cache()
        
        # Create a temporary key file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("dummy key content")
            temp_path = f.name
        
        try:
            result = is_creator_instance(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_verify_injection_allowed_without_key(self):
        """Test that verify_injection_allowed denies when no key."""
        from core.security import verify_injection_allowed, clear_creator_cache
        
        clear_creator_cache()
        
        with patch('core.security.is_creator_instance', return_value=False):
            allowed, message = verify_injection_allowed()
            assert allowed is False
            assert "denied" in message.lower()
    
    def test_verify_injection_allowed_with_key(self):
        """Test that verify_injection_allowed permits when key exists."""
        from core.security import verify_injection_allowed, clear_creator_cache
        
        clear_creator_cache()
        
        with patch('core.security.is_creator_instance', return_value=True):
            allowed, message = verify_injection_allowed()
            assert allowed is True
            assert "allowed" in message.lower()
    
    def test_get_creator_key_path(self):
        """Test that get_creator_key_path returns a valid path."""
        from core.security import get_creator_key_path
        
        path = get_creator_key_path()
        assert isinstance(path, Path)
        assert path.name == "creator_private_key.pem"
    
    def test_creator_cache_works(self):
        """Test that the cache prevents repeated filesystem checks."""
        from core.security import is_creator_instance, clear_creator_cache, _is_creator_cache
        
        clear_creator_cache()
        
        with patch('pathlib.Path.exists', return_value=True) as mock_exists:
            with patch('pathlib.Path.is_file', return_value=True):
                # First call should check filesystem
                result1 = is_creator_instance()
                # Second call should use cache
                result2 = is_creator_instance()
                
                assert result1 is True
                assert result2 is True
                # exists should only be called once due to caching
                assert mock_exists.call_count == 1
