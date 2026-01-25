import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from core.hashtag_manager import HashtagManager

@pytest.fixture
def manager():
    return HashtagManager()

@pytest.mark.asyncio
async def test_generate_hashtags_aesthetic_map_neon(manager):
    """Test neon mapping."""
    visual_spec = {"lighting": "neon lights"}
    with patch('core.hashtag_manager.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"result": "[]"}
        tags = await manager.generate_hashtags("Test", visual_spec)
        assert any("neon" in t or "cyber" in t or "light" in t for t in tags)

@pytest.mark.asyncio
async def test_generate_hashtags_aesthetic_map_glitch(manager):
    """Test glitch mapping."""
    visual_spec = {"atmosphere": "glitchy"}
    with patch('core.hashtag_manager.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"result": "[]"}
        tags = await manager.generate_hashtags("Test", visual_spec)
        assert any("glitch" in t or "data" in t for t in tags)

@pytest.mark.asyncio
async def test_generate_hashtags_aesthetic_map_pink(manager):
    """Test pink mapping."""
    visual_spec = {"color_palette": "pink"}
    with patch('core.hashtag_manager.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"result": "[]"}
        tags = await manager.generate_hashtags("Test", visual_spec)
        assert any("pink" in t or "kawaii" in t for t in tags)

@pytest.mark.asyncio
async def test_generate_hashtags_llm_integration(manager):
    """Test that LLM tags are integrated."""
    visual_spec = {"lighting": "dark"}
    mock_llm_res = {"result": '["#unique_tag_1", "unique_tag_2"]'}
    
    with patch('core.hashtag_manager.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_res
        
        tags = await manager.generate_hashtags("Test", visual_spec)
        
        assert "#unique_tag_1" in tags
        assert "#unique_tag_2" in tags
