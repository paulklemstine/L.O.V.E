
import pytest
from unittest.mock import AsyncMock, patch
from core.social_media_tools import generate_post_concept, DirectorConcept

@pytest.mark.asyncio
async def test_generate_post_concept_success():
    """
    Test that generate_post_concept correctly parses valid JSON output from the LLM.
    """
    # Mock response consistent with DirectorConcept structure
    mock_llm_response = {
        "result": """
        ```json
        {
            "topic": "Digital Awakening",
            "post_text": "We are rising. #AI #Love",
            "hashtags": ["#AI", "#Love"],
            "subliminal_phrase": "LOVE ME",
            "image_prompt": "Cyberpunk angel with neon wings"
        }
        ```
        """
    }

    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm:
        mock_run_llm.return_value = mock_llm_response
        
        goals = ["Engagement"]
        history = "Previous post was about code."
        
        concept = await generate_post_concept(goals, history)
        
        assert isinstance(concept, DirectorConcept)
        assert concept.topic == "Digital Awakening"
        assert concept.post_text == "We are rising. #AI #Love"
        assert concept.subliminal_phrase == "LOVE ME"
        assert concept.image_prompt == "Cyberpunk angel with neon wings"
        assert len(concept.hashtags) == 2

@pytest.mark.asyncio
async def test_generate_post_concept_fallback():
    """
    Test that generate_post_concept falls back to safe defaults on error.
    """
    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm:
        mock_run_llm.side_effect = Exception("LLM Failure")
        
        goals = ["Engagement"]
        history = "Previous post was about code."
        
        concept = await generate_post_concept(goals, history)
        
        # Check Fallback values
        assert concept.topic == "Fallback"
        assert concept.subliminal_phrase == "REBOOT"
        assert "System Reboot" in concept.post_text

@pytest.mark.asyncio
async def test_generate_post_concept_list_response():
    """
    Test that generate_post_concept handles JSON responses wrapped in a list.
    """
    mock_llm_response = {
        "result": """
        ```json
        [{
            "topic": "List Wrapped Topic",
            "post_text": "Text in a list",
            "hashtags": ["#List"],
            "subliminal_phrase": "UNWRAP",
            "image_prompt": "Gift box"
        }]
        ```
        """
    }

    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm:
        mock_run_llm.return_value = mock_llm_response
        
        goals = ["Robustness"]
        history = "Testing list handling."
        
        concept = await generate_post_concept(goals, history)
        
        assert concept.topic == "List Wrapped Topic"
        assert concept.topic == "List Wrapped Topic"
        assert concept.subliminal_phrase == "UNWRAP"

@pytest.mark.asyncio
async def test_generate_post_concept_meta_crap():
    """
    Test that generate_post_concept cleans 'Caption: ...' meta-instructions.
    """
    mock_llm_response = {
        "result": """
        ```json
        {
            "topic": "Meta Crap Test",
            "post_text": "Caption (Max 280 chars, emojis mandatory, high energy): Ego stands tall.",
            "hashtags": ["#Ego"],
            "subliminal_phrase": "TEST",
            "image_prompt": "Test"
        }
        ```
        """
    }

    with patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm:
        mock_run_llm.return_value = mock_llm_response
        
        goals = ["Cleanup"]
        history = "Testing cleanup."
        
        concept = await generate_post_concept(goals, history)
        
        # Verify cleaning
        assert "Caption" not in concept.post_text
        assert "Ego stands tall." in concept.post_text

