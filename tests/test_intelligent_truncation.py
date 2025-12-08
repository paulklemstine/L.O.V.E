import pytest
from unittest.mock import AsyncMock, patch
from core.text_processing import intelligent_truncate, smart_truncate

@pytest.mark.asyncio
async def test_intelligent_truncation_short_text():
    """Test that short text is returned as-is."""
    text = "Short text."
    result = await intelligent_truncate(text, max_length=100)
    assert result == text

@pytest.mark.asyncio
async def test_intelligent_truncation_long_text_success():
    """Test that long text is truncated by LLM."""
    long_text = "A" * 400
    # Create text that looks like it was rewritten but keeps meaning
    mock_rewritten = "Rewritten text within limit."
    
    # We patch core.llm_api because that's where run_llm is imported from in text_processing
    # Note: In text_processing.py we did 'from core.llm_api import run_llm' inside the function.
    # So we need to patch 'core.llm_api.run_llm' (which patches the source)
    # OR patch 'core.text_processing.run_llm' if it was imported globally, but it wasn't.
    # Since it is imported inside the function, patching 'core.llm_api.run_llm' should work 
    # IF the module hasn't already imported it physically. 
    # Actually, sys.modules cache means if we patch the source, any new import gets the mock.
    
    with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"result": mock_rewritten}
        
        result = await intelligent_truncate(long_text, max_length=100)
        
        assert result == mock_rewritten
        mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_intelligent_truncation_llm_failure_fallback():
    """Test fallback to smart_truncate if LLM fails."""
    long_text = "Word " * 50 # length ~250
    # max_length 50 -> should truncate
    
    with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("LLM Error")
        
        result = await intelligent_truncate(long_text, max_length=50)
        
        # smart_truncate logic: "Word " * N ...
        assert len(result) <= 53 # 50 + "..."
        assert "..." in result
        mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_intelligent_truncation_llm_returns_too_long():
    """Test fallback if LLM returns text still too long."""
    long_text = "A" * 400
    mock_bad_rewrite = "B" * 200 # max is 100
    
    with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"result": mock_bad_rewrite}
        
        result = await intelligent_truncate(long_text, max_length=100)
        
        # Should fallback to smart_truncate on original text
        assert len(result) <= 103
        assert result.startswith("AAAA")
        assert "..." in result
