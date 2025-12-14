import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock google.colab modules before importing core.llm_api
mock_colab = MagicMock()
mock_colab_ai = MagicMock()
sys.modules["google.colab"] = mock_colab
sys.modules["google.colab.ai"] = mock_colab_ai

# Now import the module under test
from core import llm_api

@pytest.mark.asyncio
async def test_colab_detection_and_execution():
    """
    Test that IS_COLAB is detected as True when google.colab is present,
    and that Gemini 3 is prioritized and executed via google.colab.ai.
    """
    assert llm_api.IS_COLAB is True, "IS_COLAB should be True when google.colab is mocked"

    # Test ranking prioritization
    # Initialize mock stats for gemini-3-pro-preview
    llm_api.MODEL_STATS["gemini-3-pro-preview"] = {
        "provider": "gemini", 
        "total_time_spent": 0, 
        "total_tokens_generated": 0, 
        "successful_calls": 0, 
        "failed_calls": 0,
        "reasoning_score": 100
    }
    
    # Force cooldown checks to pass
    llm_api.LLM_AVAILABILITY["gemini-3-pro-preview"] = 0
    
    ranked_models = llm_api.rank_models()
    assert "gemini-3-pro-preview" in ranked_models
    # It should likely be at the top due to the massive boost
    assert ranked_models[0] == "gemini-3-pro-preview"

    # Test execution
    mock_colab_ai.generate_text.return_value = "Colab Response"
    
    result = await llm_api.run_llm(
        prompt_text="Hello Colab", 
        force_model="gemini-3-pro-preview"
    )
    
    # Verify google.colab.ai.generate_text was called correctly
    mock_colab_ai.generate_text.assert_called_with("Hello Colab", model_name="gemini-3-pro-preview")
    assert result == "Colab Response"

if __name__ == "__main__":
    # Run the test manually if executed directly
    import asyncio
    asyncio.run(test_colab_detection_and_execution())
