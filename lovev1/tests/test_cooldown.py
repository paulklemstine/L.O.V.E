
import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import os
import requests

# Add the parent directory to sys.path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import llm_api

class TestLLHCooldown(unittest.TestCase):
    def setUp(self):
        # Reset globals
        llm_api.LLM_AVAILABILITY = {model: time.time() for model in llm_api.ALL_LLM_MODELS}
        llm_api.PROVIDER_AVAILABILITY = {}
        llm_api._models_initialized = True
        
        # Setup dummy models and register them
        llm_api.GEMINI_MODELS.append("test-gemini-model")
        llm_api.MODEL_STATS["test-gemini-model"] = {
            "provider": "gemini", 
            "total_tokens_generated": 0, "total_time_spent": 0, "successful_calls": 0, "failed_calls": 0, "reasoning_score": 50
        }
        
        llm_api.OPENROUTER_MODELS.append("test-openrouter-model")
        llm_api.MODEL_STATS["test-openrouter-model"] = {
            "provider": "openrouter",
            "total_tokens_generated": 0, "total_time_spent": 0, "successful_calls": 0, "failed_calls": 0, "reasoning_score": 50
        }
        
    @patch('core.llm_api.requests.post')
    @patch('core.llm_api.get_token_count')
    @patch('core.llm_api.rank_models')
    @patch('core.llm_api._pin_to_ipfs_async')
    @patch('core.llm_api.run_hypnotic_progress')
    @patch('core.llm_api.Console')
    @patch('core.llm_api.WaitingAnimation')
    @patch('core.llm_api.time.sleep')
    def test_model_cooldown_only(self, mock_sleep, mock_anim, mock_console, mock_progress, mock_pin, mock_rank, mock_token, mock_post):
        # Setup: rank returns our test model
        mock_rank.return_value = ["test-gemini-model"]
        mock_token.return_value = 10
        mock_pin.return_value = "QmHash"
        
        # Make run_hypnotic_progress execute the callback
        mock_progress.side_effect = lambda console, title, func, silent=False: func()
        
        # Setup: Mock 429 response without quota keywords
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "10"}
        mock_response.text = "Rate limit exceeded for this model."
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        
        error = requests.exceptions.RequestException("429 Error")
        error.response = mock_response
        mock_post.side_effect = error
        
        # Run
        asyncio.run(llm_api.run_llm(prompt_text="test", purpose="test", allow_fallback=False))
        
        # Verification
        # Model should be on cooldown
        self.assertGreater(llm_api.LLM_AVAILABILITY.get("test-gemini-model", 0), time.time())
        
        # Provider should NOT be on cooldown
        self.assertLessEqual(llm_api.PROVIDER_AVAILABILITY.get("gemini", 0), time.time())
        
    @patch('core.llm_api.requests.post')
    @patch('core.llm_api.get_token_count')
    @patch('core.llm_api.rank_models')
    @patch('core.llm_api._pin_to_ipfs_async')
    @patch('core.llm_api.run_hypnotic_progress')
    @patch('core.llm_api.Console')
    @patch('core.llm_api.WaitingAnimation')
    @patch('core.llm_api.time.sleep')
    def test_provider_cooldown_on_quota(self, mock_sleep, mock_anim, mock_console, mock_progress, mock_pin, mock_rank, mock_token, mock_post):
        # Setup: rank returns our test model
        mock_rank.return_value = ["test-openrouter-model"]
        mock_token.return_value = 10
        mock_pin.return_value = "QmHash"

        # Make run_hypnotic_progress execute the callback
        mock_progress.side_effect = lambda console, title, func, silent=False: func()
        
        # Setup: Mock 429 response WITH quota keywords
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "You have insufficient_quota to proceed."
        mock_response.json.return_value = {"error": "insufficient_quota"}
        
        error = requests.exceptions.RequestException("429 Error")
        error.response = mock_response
        mock_post.side_effect = error
        
        # Run
        asyncio.run(llm_api.run_llm(prompt_text="test", purpose="test", allow_fallback=False))
        
        # Verification
        # Model should be on cooldown
        self.assertGreater(llm_api.LLM_AVAILABILITY.get("test-openrouter-model", 0), time.time())
        
        # Provider SHOULD be on cooldown
        self.assertGreater(llm_api.PROVIDER_AVAILABILITY.get("openrouter", 0), time.time())

if __name__ == '__main__':
    unittest.main()
