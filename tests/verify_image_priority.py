import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add current directory to path so we can import core
sys.path.append(os.getcwd())

from core.image_generation_pool import generate_image_with_pool, IMAGE_MODEL_AVAILABILITY

async def test_provider_priority():
    print("Testing Provider Priority...")
    # Mock providers
    with patch('core.image_generation_pool._generate_with_pollinations') as mock_polly, \
         patch('core.image_generation_pool._generate_with_horde') as mock_horde, \
         patch('core.image_generation_pool._generate_with_stability') as mock_stability, \
         patch('core.logging.log_event'):

        # Scenario 1: Pollinations succeeds
        mock_polly.return_value = "pollinations_image"
        
        result = await generate_image_with_pool("test prompt")
        
        assert result == "pollinations_image"
        mock_polly.assert_called_once()
        mock_horde.assert_not_called()
        mock_stability.assert_not_called()
    print("Provider Priority: PASS")

async def test_provider_fallback():
    print("Testing Provider Fallback...")
    # Mock providers
    with patch('core.image_generation_pool._generate_with_pollinations') as mock_polly, \
         patch('core.image_generation_pool._generate_with_horde') as mock_horde, \
         patch('core.image_generation_pool._generate_with_stability') as mock_stability, \
         patch('core.logging.log_event'):

        # Scenario 2: Pollinations fails, Horde succeeds
        mock_polly.side_effect = Exception("Pollinations failed")
        mock_horde.return_value = "horde_image"
        
        result = await generate_image_with_pool("test prompt")
        
        assert result == "horde_image"
        mock_polly.assert_called_once()
        mock_horde.assert_called_once()
        mock_stability.assert_not_called()
    print("Provider Fallback: PASS")

if __name__ == "__main__":
    # Reset cooldowns for testing
    IMAGE_MODEL_AVAILABILITY.clear()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_provider_priority())
        loop.run_until_complete(test_provider_fallback())
        print("ALL TESTS PASSED")
    finally:
        loop.close()
