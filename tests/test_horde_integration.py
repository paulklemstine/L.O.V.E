import pytest
from unittest.mock import patch, MagicMock
from core.llm_api import run_llm
from core.image_api import generate_image
from PIL import Image
import io

@patch('core.llm_api.GEMINI_MODELS', [])
@patch('core.llm_api.LOCAL_MODELS_CONFIG', [])
@patch('requests.get')
@patch('requests.post')
def test_horde_text_generation(mock_post, mock_get):
    """
    Tests that the AI Horde text generation is working.
    """
    # Mock the model fetching
    mock_get.return_value.raise_for_status = MagicMock()
    mock_get.return_value.json = MagicMock(return_value=[{"name": "Mythalion-13B", "performance": 100}])

    # Mock the async job submission
    mock_post.return_value.raise_for_status = MagicMock()
    mock_post.return_value.json = MagicMock(return_value={"id": "test_job_id"})

    # Mock the status check
    mock_get.return_value.raise_for_status = MagicMock()
    mock_get.return_value.json = MagicMock(return_value={"done": True, "generations": [{"text": "test response"}]})

    result = run_llm("test prompt", purpose="general")
    assert result['result'] == "test response"

@patch('requests.get')
@patch('requests.post')
def test_horde_image_generation(mock_post, mock_get):
    """
    Tests that the AI Horde image generation is working.
    """
    # Mock the async job submission
    mock_post.return_value.raise_for_status = MagicMock()
    mock_post.return_value.json = MagicMock(return_value={"id": "test_job_id"})

    # Mock the model fetching, status check, and image download
    mock_model_response = MagicMock()
    mock_model_response.raise_for_status = MagicMock()
    mock_model_response.json.return_value = [{"name": "stable_diffusion_2.1", "performance": 100}]

    mock_status_response = MagicMock()
    mock_status_response.raise_for_status = MagicMock()
    mock_status_response.json.return_value = {"done": True, "generations": [{"img": "http://example.com/test.png"}]}

    mock_image_response = MagicMock()
    mock_image_response.raise_for_status = MagicMock()
    mock_image_response.content = b'test_image_data'

    # The first call to get() is for the models, the second for status, the third for the image
    mock_get.side_effect = [mock_model_response, mock_status_response, mock_image_response]

    with patch('PIL.Image.open', MagicMock()) as mock_open:
        generate_image("test prompt")
        mock_open.assert_called_once()
