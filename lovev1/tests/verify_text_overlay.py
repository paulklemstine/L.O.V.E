import asyncio
import sys
import os
from unittest.mock import MagicMock, patch
from PIL import Image

# Add current directory to path
sys.path.append(os.getcwd())

from core.image_generation_pool import generate_image_with_pool, IMAGE_MODEL_AVAILABILITY

async def test_overlay_fallback():
    print("Testing Text Overlay Fallback Logic...")
    
    # Mock providers and overlay utils
    with patch('core.image_generation_pool._generate_with_pollinations') as mock_polly, \
         patch('core.image_generation_pool._generate_with_horde') as mock_horde, \
         patch('core.image_generation_pool._generate_with_stability') as mock_stability, \
         patch('core.image_generation_pool.overlay_text_on_image') as mock_overlay, \
         patch('core.logging.log_event'):

        # Scenario 1: Pollinations succeeds -> YES overlay (Now enforced)
        mock_polly.return_value = Image.new("RGB", (100, 100))
        
        await generate_image_with_pool("prompt", text_content="TEST")
        
        # We now verify that overlay IS called even for Pollinations
        assert mock_overlay.call_count >= 1
        print("Scenario 1 (Pollinations + Manual Overlay): PASS")

        # Scenario 2: Pollinations fails, Horde succeeds -> YES overlay
        mock_polly.side_effect = Exception("Fail")
        mock_horde.return_value = Image.new("RGB", (100, 100))
        
        await generate_image_with_pool("prompt", text_content="TEST")
        
        mock_overlay.assert_called_once()

        print("Scenario 2 (Fallback): PASS")

if __name__ == "__main__":
    IMAGE_MODEL_AVAILABILITY.clear()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_overlay_fallback())
        print("ALL TESTS PASSED")
    finally:
        loop.close()
