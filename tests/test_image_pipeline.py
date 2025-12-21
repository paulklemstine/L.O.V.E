
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
from core.image_generation_pool import generate_image_with_pool

# Create a dummy image for testing
@pytest.fixture
def dummy_image():
    return Image.new('RGB', (100, 100), color='red')

@pytest.mark.asyncio
async def test_pollinations_programmatic_overlay(dummy_image):
    """
    Story 2.1 (Revised): Verify Pollinations receives CLEAN prompt and DOES manual overlay.
    """
    with patch('core.image_generation_pool._generate_with_pollinations', new_callable=AsyncMock) as mock_poly, \
         patch('core.image_generation_pool.overlay_text_on_image') as mock_overlay, \
         patch('core.image_generation_pool.IMAGE_MODEL_AVAILABILITY', {}): # Ensure no cooldowns
        
        mock_poly.return_value = dummy_image
        mock_overlay.return_value = dummy_image
        
        prompt = "A beautiful sunset"
        text = "WAKE UP"
        
        # force_provider='pollinations' to isolate the test
        await generate_image_with_pool(prompt, text_content=text, force_provider='pollinations')
        
        # Assertions
        # 1. Verify CLEAN prompt (no embedding instruction)
        args, _ = mock_poly.call_args
        called_prompt = args[0]
        assert called_prompt == "A beautiful sunset"
        assert "is written in neon light style" not in called_prompt
        
        # 2. Verify manual overlay IS called
        mock_overlay.assert_called_once()
        call_args = mock_overlay.call_args
        assert call_args[0][1] == "WAKE UP"

@pytest.mark.asyncio
async def test_horde_fallback_overlay(dummy_image):
    """
    Story 2.2 & 2.3: Verify Horde receives CLEAN prompt and DOES manual overlay.
    """
    with patch('core.image_generation_pool._generate_with_pollinations', new_callable=AsyncMock) as mock_poly, \
         patch('core.image_generation_pool._generate_with_horde', new_callable=AsyncMock) as mock_horde, \
         patch('core.image_generation_pool.overlay_text_on_image') as mock_overlay, \
         patch('core.image_generation_pool.IMAGE_MODEL_AVAILABILITY', {}):
        
        # Pollinations fails
        mock_poly.side_effect = Exception("Pollinations Down")
        # Horde succeeds
        mock_horde.return_value = dummy_image
        # Overlay returns image
        mock_overlay.return_value = dummy_image
        
        prompt = "A beautiful sunset"
        text = "OBEY"
        
        # We don't force provider, letting the pool logic flow: Poly -> Horde
        await generate_image_with_pool(prompt, text_content=text)
        
        # Assertions
        
        # 1. Verify Pollinations was attempted (and failed)
        mock_poly.assert_called_once()
        
        # 2. Verify Horde was called with CLEAN prompt
        args, _ = mock_horde.call_args
        called_prompt = args[0]
        assert called_prompt == "A beautiful sunset"
        assert "OBEY" not in called_prompt # Ensure text wasn't embedded
        
        # 3. Verify Overlay WAS called
        mock_overlay.assert_called_once()
        # Verify arguments pass to overlay
        call_args = mock_overlay.call_args
        assert call_args[0][1] == "OBEY"  # text argument
