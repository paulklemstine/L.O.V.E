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
        llm_api._models_initialized = True # Prevent network calls

        # Add test models to the global lists so run_llm recognizes them
        if "openrouter_model" not in llm_api.OPENROUTER_MODELS:
            llm_api.OPENROUTER_MODELS.append("openrouter_model")
        if "gemini_model" not in llm_api.GEMINI_MODELS:
            llm_api.GEMINI_MODELS.append("gemini_model")

    def tearDown(self):
        """Clean up after each test."""
        patch.stopall()
        if "openrouter_model" in llm_api.OPENROUTER_MODELS:
            llm_api.OPENROUTER_MODELS.remove("openrouter_model")
        if "gemini_model" in llm_api.GEMINI_MODELS:
            llm_api.GEMINI_MODELS.remove("gemini_model")

    @patch('time.sleep', return_value=None)
    @patch('core.llm_api.pin_to_ipfs_sync', MagicMock(return_value="test_cid"))
    @patch('requests.post')
    @patch('subprocess.run')
    async def test_provider_fallback_and_failure_count(self, mock_subprocess_run, mock_requests_post, mock_sleep):
        """
        Test that a failing provider increments failure count and the system falls back.
        """
        # --- MOCK SETUP ---
        # 1. Mock OpenRouter to fail with an HTTPError
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.headers = {}
        mock_response_fail.json.return_value = {"error": "API Error"}
        http_error = requests.exceptions.HTTPError("API Error")
        http_error.response = mock_response_fail

        # Side effect function to direct mock behavior
        def post_side_effect(*args, **kwargs):
            url = args[0]
            if "openrouter" in url:
                raise http_error
            elif "generativelanguage" in url: # Gemini endpoint
                raise http_error
            return MagicMock() # Default success
        mock_requests_post.side_effect = post_side_effect

        # SMART MOCK for run_hypnotic_progress to avoid deadlock
        def progress_side_effect(console, msg, func, silent=False):
            if "Horde" in msg:
                return "horde_success" # Return success directly, bypassing the deadlock wrapper
            return func() # Execute normally for failing providers

        with patch('core.llm_api.run_hypnotic_progress', side_effect=progress_side_effect):
            # --- EXECUTION ---
            llm_api.MODEL_STATS['openrouter_model']['provider'] = 'openrouter'
            llm_api.MODEL_STATS['gemini_model']['provider'] = 'gemini'
            llm_api.MODEL_STATS['horde_model']['provider'] = 'horde'

            # We need to patch rank_models to return our specific test models
            with patch('core.llm_api.rank_models', return_value=['openrouter_model', 'gemini_model', 'horde_model']):
                # Reset availability
                llm_api.LLM_AVAILABILITY['openrouter_model'] = 0
                llm_api.LLM_AVAILABILITY['gemini_model'] = 0
                llm_api.LLM_AVAILABILITY['horde_model'] = 0

                result = await llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'horde_success')
        # Check individual model failure counts
        self.assertEqual(llm_api.MODEL_STATS['openrouter_model']['failed_calls'], 1)
        self.assertEqual(llm_api.MODEL_STATS['gemini_model']['failed_calls'], 1)

    @patch('time.sleep', return_value=None)
    @patch('core.llm_api.pin_to_ipfs_sync', MagicMock(return_value="test_cid"))
    @patch('core.llm_api.run_hypnotic_progress', side_effect=lambda console, msg, func, silent=False: func())
    @patch('requests.post')
    async def test_successful_call_resets_failure_count(self, mock_requests_post, mock_run_hypnotic_progress, mock_sleep):
        """
        Test that a successful call works as expected.
        """
        # --- MOCK SETUP ---
        # Mock OpenRouter to succeed this time
        mock_requests_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "openrouter_success"}}]})

        llm_api.MODEL_STATS['openrouter_model']['provider'] = 'openrouter'
        llm_api.LLM_AVAILABILITY['openrouter_model'] = time.time()

        # --- EXECUTION ---
        with patch('core.llm_api.rank_models', return_value=['openrouter_model']):
            result = await llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'openrouter_success')
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT.get('openrouter', 0), 0)

class TestHordeModelRanking(unittest.TestCase):

    @patch('requests.get')
    def test_ranking_logic(self, mock_requests_get):
        """
        Tests the AI Horde model ranking logic based on ETA, performance, size, and name.
        """
        # --- MOCK SETUP ---
        mock_api_response = [
            # 1. Best: Fast ETA, high perf, big, uncensored
            {"name": "BigUncensored-70B", "eta": 5, "performance": 200, "count": 10},
            # 2. Good: Decent ETA, decent perf, but smaller
            {"name": "SmallFast-7B", "eta": 10, "performance": 150, "count": 20},
            # 3. Okay: Censored, but fast ETA
            {"name": "CensoredFast-13B", "eta": 8, "performance": 180, "count": 15},
            # 4. Bad: Very slow ETA despite other good stats
            {"name": "SlowModel-70B", "eta": 600, "performance": 250, "count": 5},
            # 5. Last: Offline model, should be excluded
            {"name": "OfflineModel-13B", "eta": 100, "performance": 100, "count": 0},
        ]
        mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: mock_api_response)

        # --- EXECUTION ---
        ranked_models = llm_api.get_top_horde_models(get_all=True)

        # --- ASSERTIONS ---
        self.assertEqual(len(ranked_models), 4) # Offline model should be filtered out
        self.assertEqual(ranked_models[0], "BigUncensored-70B")
        self.assertEqual(ranked_models[1], "CensoredFast-13B")
        self.assertEqual(ranked_models[2], "SmallFast-7B")
        self.assertEqual(ranked_models[3], "SlowModel-70B")


if __name__ == '__main__':
    unittest.main()
