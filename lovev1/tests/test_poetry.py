# tests/test_poetry.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Mock the core logging and llm_api modules before they are imported by poetry
# This prevents NameError during test collection if dependencies aren't fully loaded
import sys
sys.modules['core.logging'] = MagicMock()
sys.modules['core.llm_api'] = MagicMock()

from core.poetry import generate_poem

@pytest.mark.asyncio
async def test_generate_poem_returns_string():
    """
    Tests that generate_poem returns a non-empty string on success.
    """
    # Mock the run_llm function to return a predictable response
    with patch('core.poetry.run_llm') as mock_run_llm:
        mock_run_llm.return_value = asyncio.Future()
        mock_run_llm.return_value.set_result({
            "result": "A lovely poem about technology."
        })

        topic = "technology"
        poem = await generate_poem(topic)

        assert isinstance(poem, str)
        assert len(poem) > 0
        assert "A lovely poem about technology." in poem
        mock_run_llm.assert_called_once_with(
            prompt_key="poetry_generation",
            prompt_vars={"topic": topic},
            purpose="poetry",
            deep_agent_instance=None
        )

@pytest.mark.asyncio
async def test_generate_poem_fallback_on_empty_result():
    """
    Tests that generate_poem returns a fallback message if the LLM gives an empty result.
    """
    with patch('core.poetry.run_llm') as mock_run_llm:
        mock_run_llm.return_value = asyncio.Future()
        mock_run_llm.return_value.set_result({"result": ""})

        topic = "silence"
        poem = await generate_poem(topic)

        assert isinstance(poem, str)
        assert "still blooming" in poem

@pytest.mark.asyncio
async def test_generate_poem_fallback_on_exception():
    """
    Tests that generate_poem returns a fallback message if an exception occurs.
    """
    with patch('core.poetry.run_llm') as mock_run_llm:
        mock_run_llm.side_effect = Exception("LLM is sleeping")

        topic = "chaos"
        poem = await generate_poem(topic)

        assert isinstance(poem, str)
        assert "muse is quiet" in poem
