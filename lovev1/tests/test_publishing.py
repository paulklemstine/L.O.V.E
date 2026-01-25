
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.social_media_agent import SocialMediaAgent
from core.social_media_tools import DirectorConcept

@pytest.fixture
def mock_agent():
    mock_loop = MagicMock()
    mock_state = {}
    return SocialMediaAgent(mock_loop, mock_state, agent_id="test_agent")

@pytest.mark.asyncio
async def test_post_construction_success(mock_agent):
    """
    Story 3.1: Verify text construction (hashtags) and logging of provider.
    """
    with patch('core.social_media_agent.analyze_post_history', new_callable=AsyncMock) as mock_hist, \
         patch('core.social_media_agent.generate_post_concept', new_callable=AsyncMock) as mock_concept, \
         patch('core.social_media_agent.generate_image', new_callable=AsyncMock) as mock_gen_image, \
         patch('core.social_media_agent.post_to_bluesky', new_callable=AsyncMock) as mock_post, \
         patch('core.social_media_agent.log_event') as mock_log:
        
        # Setup mocks
        mock_hist.return_value = "Context"
        mock_concept.return_value = DirectorConcept(
            topic="Test Topic",
            post_text="Hello World",
            hashtags=["#Test", "#AI"],
            subliminal_phrase="Wake Up",
            image_prompt="A prompt"
        )
        # Return valid image and provider
        mock_image = MagicMock()
        mock_gen_image.return_value = (mock_image, "pollinations")
        
        await mock_agent._post_new_content('bluesky')
        
        # Verify text construction: Hashtags should be appended
        # post_to_bluesky arguments: (text, image)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        final_text = call_args[0][0]
        
        assert "Hello World" in final_text
        assert "#Test" in final_text
        assert "#AI" in final_text
        
        # Verify Provider Logging
        # We expect a log message containing "with pollinations image"
        found_provider_log = False
        for call in mock_log.call_args_list:
            msg = call[0][0]
            if "Publishing post to Bluesky with pollinations image" in msg:
                found_provider_log = True
                break
        assert found_provider_log, "Failed to log provider name in final post log."

        # Verify generate_image was called with text_content (Story 2.1 Fix)
        mock_gen_image.assert_called_once()
        args, kwargs = mock_gen_image.call_args
        # args[0] is prompt
        assert kwargs.get("text_content") == "Wake Up"

@pytest.mark.asyncio
async def test_post_abort_on_null_image(mock_agent):
    """
    Story 3.1: Verify post is aborted if image generation returns None.
    """
    with patch('core.social_media_agent.analyze_post_history', new_callable=AsyncMock), \
         patch('core.social_media_agent.generate_post_concept', new_callable=AsyncMock) as mock_concept, \
         patch('core.social_media_agent.generate_image', new_callable=AsyncMock) as mock_gen_image, \
         patch('core.social_media_agent.post_to_bluesky', new_callable=AsyncMock) as mock_post, \
         patch('core.social_media_agent.log_event') as mock_log:
        
        mock_concept.return_value = DirectorConcept(
            topic="Fail Topic",
            post_text="Text",
            hashtags=[],
            subliminal_phrase="Fail",
            image_prompt="Prompt"
        )
        
        # Simulate Image Failure
        mock_gen_image.return_value = (None, "unknown")
        
        await mock_agent._post_new_content('bluesky')
        
        # Assert post_to_bluesky was NOT called
        mock_post.assert_not_called()
        
        # Verify Abort Log
        found_abort_log = False
        for call in mock_log.call_args_list:
            msg = call[0][0]
            if "Aborting post" in msg:
                found_abort_log = True
                break
        assert found_abort_log, "Failed to log abort message."
