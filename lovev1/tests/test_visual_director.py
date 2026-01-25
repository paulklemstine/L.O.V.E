import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.visual_director import VisualDirector

@pytest.fixture
def director():
    return VisualDirector()

@pytest.mark.asyncio
async def test_direct_scene_valid_output(director):
    """Test that direct_scene returns a dictionary with required keys."""
    # Mock run_llm to return a valid JSON string
    mock_response = {
        "result": '''
        ```json
        {
            "subject": "A glowing cyberpunk angel",
            "lighting": "Neon blue rim lights",
            "camera_angle": "Low angle",
            "composition": "Centered",
            "color_palette": "Blue, Pink, Black",
            "atmosphere": "Misty"
        }
        ```
        '''
    }
    
    with patch('core.visual_director.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        
        result = await director.direct_scene("Cyber Angel")
        
        assert isinstance(result, dict)
        assert "subject" in result
        assert "lighting" in result
        assert result["lighting"] == "Neon blue rim lights"

@pytest.mark.asyncio
async def test_direct_scene_fallback(director):
    """Test that direct_scene returns fallback on failure."""
    with patch('core.visual_director.run_llm', side_effect=Exception("LLM Fail")):
        result = await director.direct_scene("Cyber Angel")
        
        assert isinstance(result, dict)
        assert "subject" in result
        assert "Default" not in result["subject"] # Fallback uses specific logic

def test_synthesize_image_prompt(director):
    """Test that synthesize_image_prompt produces a string with key elements."""
    visual_spec = {
        "subject": "Test Subject",
        "lighting": "Test Lighting",
        "atmosphere": "Test Atmosphere",
        "camera_angle": "Test Angle",
        "composition": "Test Comp",
        "color_palette": "Test Colors"
    }
    subliminal = "OBEY"
    
    prompt = director.synthesize_image_prompt(visual_spec, subliminal)
    
    assert "Test Subject" in prompt
    assert "Test Lighting" in prompt
    assert "OBEY" in prompt
    assert "8k" in prompt
