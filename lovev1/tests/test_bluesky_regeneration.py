import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from core.social_media_tools import regenerate_shorter_content, post_to_bluesky

@pytest.mark.asyncio
async def test_regenerate_shorter_content():
    """Test that the regeneration function calls the LLM correctly."""
    original_text = "This is a very long post " * 20  # Clearly > 300 chars
    max_length = 300
    
    # Mock LLM response
    mock_response = {
        "result": "This is a shorter version. #Love"
    }
    
    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm:
        mock_run_llm.return_value = mock_response
        
        result = await regenerate_shorter_content(original_text, max_length)
        
        assert result == "This is a shorter version. #Love"
        
        # Verify prompt details
        call_args = mock_run_llm.call_args
        prompt = call_args[0][0]
        assert f"UNDER {max_length}" in prompt
        assert str(len(original_text)) in prompt

@pytest.mark.asyncio
async def test_post_to_bluesky_regeneration_loop():
    """Test that post_to_bluesky attempts to regenerate when text is too long."""
    long_text = "A" * 350
    short_text = "A" * 100
    
    # Mock dependencies
    with patch('core.social_media_tools.post_to_bluesky_with_image') as mock_post_api, \
         patch('core.social_media_tools.regenerate_shorter_content', new_callable=AsyncMock) as mock_regenerate:
        
        # Setup mock behavior: first call returns short enough text
        mock_regenerate.return_value = short_text
        mock_post_api.return_value = {"uri": "test-uri"}
        
        result = await post_to_bluesky(long_text)
        
        # Verify regeneration was called
        mock_regenerate.assert_called_once()
        
        # Verify API was called with the SHORT text, not the long one
        mock_post_api.assert_called_with(short_text, None)
        assert result == {"uri": "test-uri"}

@pytest.mark.asyncio
async def test_post_to_bluesky_failure_after_retries():
    """Test that it fails gracefully if regeneration keeps returning long text."""
    long_text = "A" * 350
    
    with patch('core.social_media_tools.regenerate_shorter_content', new_callable=AsyncMock) as mock_regenerate:
        # Always return long text
        mock_regenerate.return_value = long_text
        
        result = await post_to_bluesky(long_text)
        
        assert "Error" in str(result)
        assert "attempts" in str(result)
        assert mock_regenerate.call_count == 3  # Should retry 3 times
