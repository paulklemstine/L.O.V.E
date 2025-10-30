import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import time
import sys
import os
import requests
import subprocess
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import llm_api

class TestLLMApi(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up test environment before each test."""
        llm_api.PROVIDER_FAILURE_COUNT = {provider: 0 for provider in ["gemini", "horde", "openrouter"]}
        llm_api.LLM_AVAILABILITY = {model: time.time() for model in llm_api.ALL_LLM_MODELS}

    def tearDown(self):
        """Clean up after each test."""
        patch.stopall()

    @patch('core.llm_api.pin_to_ipfs_sync', MagicMock(return_value="test_cid"))
    @patch('core.llm_api.run_hypnotic_progress', side_effect=lambda console, msg, func, silent=False: func())
    @patch('requests.post')
    @patch('subprocess.run')
    async def test_provider_fallback_and_failure_count(self, mock_subprocess_run, mock_requests_post):
        """
        Test that a failing provider increments failure count and the system falls back.
        """
        # --- MOCK SETUP ---
        # 1. Mock OpenRouter to fail with an HTTPError
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.headers = {}
        http_error = requests.exceptions.HTTPError("API Error")
        http_error.response = mock_response_fail

        # 2. Mock Gemini to fail with a CalledProcessError
        subprocess_error = subprocess.CalledProcessError(1, "cmd", stderr="Gemini error")

        # 3. Mock Horde to succeed
        mock_horde_success_response = MagicMock()
        mock_horde_success_response.status_code = 200
        mock_horde_success_response.json.return_value = {"id": "horde_job_id"}

        # Side effect function to direct mock behavior
        def post_side_effect(*args, **kwargs):
            url = args[0]
            if "openrouter" in url:
                raise http_error
            elif "horde" in url:
                return mock_horde_success_response
            return MagicMock() # Default success for other unexpected calls
        mock_requests_post.side_effect = post_side_effect
        mock_subprocess_run.side_effect = subprocess_error

        # Mock the GET request for Horde's status check to succeed
        with patch('requests.get') as mock_requests_get:
            mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: {"done": True, "generations": [{"text": "horde_success"}]})

            # --- EXECUTION ---
            # Use a patch to control provider order for predictability
            with patch('random.shuffle', side_effect=lambda x: x.sort(key=lambda p: ['openrouter', 'gemini', 'horde'].index(p))):
                result = await llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'horde_success')
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['openrouter'], 1)
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['gemini'], 1)
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['horde'], 0) # Resets on success

    @patch('core.llm_api.pin_to_ipfs_sync', MagicMock(return_value="test_cid"))
    @patch('core.llm_api.run_hypnotic_progress', side_effect=lambda console, msg, func, silent=False: func())
    @patch('requests.post')
    async def test_successful_call_resets_failure_count(self, mock_requests_post):
        """
        Test that a successful call resets a provider's failure count.
        """
        # --- MOCK SETUP ---
        # Set a pre-existing failure count for OpenRouter
        llm_api.PROVIDER_FAILURE_COUNT['openrouter'] = 3

        # Mock OpenRouter to succeed this time
        mock_requests_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "openrouter_success"}}]})

        # --- EXECUTION ---
        with patch('random.shuffle', side_effect=lambda x: x.sort(key=lambda p: p == 'openrouter', reverse=True)):
            result = await llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'openrouter_success')
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['openrouter'], 0)

if __name__ == '__main__':
    unittest.main()
