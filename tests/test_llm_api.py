import unittest
from unittest.mock import patch, MagicMock, call
import time
import sys
import os
import requests
import subprocess

# Ensure the 'core' directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import llm_api

# Mock the llama_cpp module to avoid ImportError during tests
sys.modules['llama_cpp'] = MagicMock()


class TestLLMApi(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Reset all global states in llm_api before each test
        llm_api.LLM_AVAILABILITY = {model: time.time() for model in llm_api.ALL_LLM_MODELS}
        llm_api.LLM_FAILURE_COUNT = {model: 0 for model in llm_api.ALL_LLM_MODELS}
        llm_api.PROVIDER_FAILURE_COUNT = {provider: 0 for provider in ["local", "gemini", "horde", "openrouter"]}
        llm_api.local_llm_instance = None

        # Mock external dependencies
        self.patch_console = patch('core.llm_api.Console', return_value=MagicMock())
        self.patch_console.start()

        self.patch_pin_to_ipfs = patch('core.llm_api.pin_to_ipfs_sync', return_value="test_cid")
        self.patch_pin_to_ipfs.start()

        # This mock will now execute the function passed to it, simulating the real behavior
        self.patch_run_hypnotic_progress = patch('core.llm_api.run_hypnotic_progress', side_effect=lambda console, msg, func, silent=False: func())
        self.patch_run_hypnotic_progress.start()

        # Mock subprocess to fail by default for Gemini calls
        self.mock_subprocess_run = patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"))
        self.mock_subprocess_run.start()

        # Mock hf_hub_download to fail, preventing local and emergency models from succeeding
        self.mock_hf_download = patch('core.llm_api.hf_hub_download', side_effect=Exception("Download failed"))
        self.mock_hf_download.start()


    def tearDown(self):
        """Clean up after each test."""
        patch.stopall()

    @patch('time.sleep', return_value=None)
    @patch('requests.get')
    @patch('requests.post')
    @patch('core.llm_api.random.shuffle')
    def test_provider_fallback_logic(self, mock_shuffle, mock_requests_post, mock_requests_get, mock_sleep):
        """
        Test that the system tries up to 3 models from one provider, then switches to another.
        """
        # --- MOCK SETUP ---
        # Define provider order: openrouter (fail) -> gemini (fail) -> horde (succeed) -> local (never reached)
        def shuffle_order(providers):
            fixed_order = ['openrouter', 'gemini', 'horde', 'local']
            providers.sort(key=lambda p: fixed_order.index(p))
        mock_shuffle.side_effect = shuffle_order

        # Create a proper HTTPError with a mock response
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.headers = {}
        http_error = requests.exceptions.HTTPError("API Error")
        http_error.response = mock_response_fail

        # Use a function for the side_effect to handle calls logically
        def mock_post_side_effect(*args, **kwargs):
            url = args[0]
            if "openrouter" in url:
                raise http_error
            elif "horde" in url:
                return MagicMock(status_code=200, json=lambda: {"id": "horde_job_id"})
            # This will catch any unexpected post requests
            raise ValueError(f"Unexpected POST request to {url}")

        mock_requests_post.side_effect = mock_post_side_effect
        mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: {"done": True, "generations": [{"text": "horde_success"}]})

        # --- EXECUTION ---
        result = llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'horde_success')
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['openrouter'], 1)
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['gemini'], 1)
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['horde'], 0) # Resets on success
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['local'], 0) # Should not be reached

    @patch('time.sleep', return_value=None)
    @patch('requests.post')
    @patch('core.llm_api.random.shuffle')
    def test_exponential_backoff_for_provider(self, mock_shuffle, mock_requests_post, mock_sleep):
        """
        Test that a failing provider is put on an exponentially increasing cooldown.
        """
        # --- MOCK SETUP ---
        # Make 'openrouter' the first provider to be tried
        def shuffle_order(providers):
            fixed_order = ['openrouter', 'gemini', 'horde', 'local']
            providers.sort(key=lambda p: fixed_order.index(p))
        mock_shuffle.side_effect = shuffle_order

        # Create a proper HTTPError with a mock response
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.headers = {}
        http_error = requests.exceptions.HTTPError("API Error")
        http_error.response = mock_response_fail

        # Make all API calls fail
        mock_requests_post.side_effect = http_error

        # --- 1st EXECUTION (First Failure) ---
        llm_api.run_llm("test prompt 1")

        # --- 1st ASSERTIONS ---
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['openrouter'], 1)
        # Cooldown should be 60 * (2**0) = 60s
        first_cooldown_end_time = min(v for k, v in llm_api.LLM_AVAILABILITY.items() if k in llm_api.OPENROUTER_MODELS)
        self.assertGreater(first_cooldown_end_time, time.time() + 59)

        # --- 2nd EXECUTION (Second Failure) ---
        # Manually fast-forward time to get past the cooldown for the openrouter provider only
        for model in llm_api.OPENROUTER_MODELS:
            llm_api.LLM_AVAILABILITY[model] = time.time()

        # Reset failure count for other providers that "failed" in the first run
        llm_api.PROVIDER_FAILURE_COUNT['gemini'] = 0
        llm_api.PROVIDER_FAILURE_COUNT['horde'] = 0
        llm_api.PROVIDER_FAILURE_COUNT['local'] = 0


        llm_api.run_llm("test prompt 2")

        # --- 2nd ASSERTIONS ---
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['openrouter'], 2)
        # Cooldown should be 60 * (2**1) = 120s
        second_cooldown_end_time = min(v for k, v in llm_api.LLM_AVAILABILITY.items() if k in llm_api.OPENROUTER_MODELS)
        self.assertGreater(second_cooldown_end_time, time.time() + 119)

    @patch('time.sleep', return_value=None)
    @patch('requests.get')
    @patch('requests.post')
    @patch('core.llm_api.random.shuffle')
    def test_successful_call_resets_provider_failure_count(self, mock_shuffle, mock_requests_post, mock_requests_get, mock_sleep):
        """
        Test that a successful call resets the provider's failure count to 0.
        """
        # --- MOCK SETUP ---
        # Make 'horde' the first provider to be tried
        def shuffle_order(providers):
            providers.sort(key=lambda p: p == 'horde', reverse=True)
        mock_shuffle.side_effect = shuffle_order

        # Set a pre-existing failure count
        llm_api.PROVIDER_FAILURE_COUNT['horde'] = 5

        # Make the call succeed
        mock_requests_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "horde_job_id"})
        mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: {"done": True, "generations": [{"text": "horde_success"}]})

        # --- EXECUTION ---
        result = llm_api.run_llm("test prompt")

        # --- ASSERTIONS ---
        self.assertEqual(result['result'], 'horde_success')
        self.assertEqual(llm_api.PROVIDER_FAILURE_COUNT['horde'], 0)


if __name__ == '__main__':
    unittest.main()
