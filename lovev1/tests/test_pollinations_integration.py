import pytest
from unittest.mock import patch, MagicMock
from core.image_generation_pool import _generate_with_pollinations
from PIL import Image
import io

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
async def test_pollinations_generation(mock_get):
    """
    Tests that the Pollinations image generation is working.
    """
    # Mock the response
    mock_response = MagicMock()
    mock_response.status = 200
    
    # Make read() awaitable
    async def async_read():
        return b'fake_image_data'
    mock_response.read = async_read
    
    # Setup the async context manager mock
    mock_get.return_value.__aenter__.return_value = mock_response
    
    with patch('PIL.Image.open', MagicMock()) as mock_open:
        await _generate_with_pollinations("test prompt")
        
        # Verify URL construction
        args, _ = mock_get.call_args
        url = args[0]
        assert "https://image.pollinations.ai/prompt/test%20prompt" in url
        assert "width=1200" in url
        assert "height=200" in url
        assert "nologo=true" in url
        assert "safe=false" in url
        
        mock_open.assert_called_once()
