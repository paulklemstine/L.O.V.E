"""
Unit tests for the Model Quality Controller.

Tests the quality tracking, automatic blacklisting, and LLM judge evaluation
functionality.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import json
import tempfile
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import model_quality_controller as mqc


class TestModelQualityController(unittest.TestCase):
    """Tests for the ModelQualityController class."""

    def setUp(self):
        """Set up test environment with fresh controller."""
        # Reset the singleton
        mqc._quality_controller = None
        
        # Use a temporary file for the blacklist
        self.temp_blacklist = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_blacklist.write('{}')
        self.temp_blacklist.close()
        
        # Patch the blacklist path
        self.original_path = mqc.AUTO_BLACKLIST_PATH
        mqc.AUTO_BLACKLIST_PATH = self.temp_blacklist.name
        
        # Create fresh controller
        self.controller = mqc.ModelQualityController()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original path
        mqc.AUTO_BLACKLIST_PATH = self.original_path
        mqc._quality_controller = None
        
        # Clean up temp file
        try:
            os.unlink(self.temp_blacklist.name)
        except:
            pass

    def test_record_success_tracks_correctly(self):
        """Test that successful calls are tracked correctly."""
        self.controller.record_success("test-model:free", "openrouter")
        
        stats = self.controller.get_model_stats("test-model:free", "openrouter")
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["successful_calls"], 1)
        self.assertEqual(stats["failed_calls"], 0)
        self.assertEqual(stats["consecutive_failures"], 0)

    def test_record_failure_tracks_correctly(self):
        """Test that failed calls are tracked correctly."""
        self.controller.record_failure("test-model:free", "openrouter", "general")
        
        stats = self.controller.get_model_stats("test-model:free", "openrouter")
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["successful_calls"], 0)
        self.assertEqual(stats["failed_calls"], 1)
        self.assertEqual(stats["consecutive_failures"], 1)

    def test_consecutive_failures_trigger_blacklist(self):
        """Test that 5 consecutive failures trigger blacklisting."""
        model_id = "bad-model:free"
        
        for i in range(4):
            blacklisted = self.controller.record_failure(model_id, "openrouter", "general")
            self.assertFalse(blacklisted)
            self.assertFalse(self.controller.is_blacklisted(model_id, "openrouter"))
        
        # 5th failure should trigger blacklist
        blacklisted = self.controller.record_failure(model_id, "openrouter", "general")
        self.assertTrue(blacklisted)
        self.assertTrue(self.controller.is_blacklisted(model_id, "openrouter"))

    def test_success_resets_consecutive_failures(self):
        """Test that a success resets the consecutive failure count."""
        model_id = "flaky-model:free"
        
        # Record 4 failures
        for _ in range(4):
            self.controller.record_failure(model_id, "openrouter", "general")
        
        stats = self.controller.get_model_stats(model_id, "openrouter")
        self.assertEqual(stats["consecutive_failures"], 4)
        
        # Record a success
        self.controller.record_success(model_id, "openrouter")
        
        stats = self.controller.get_model_stats(model_id, "openrouter")
        self.assertEqual(stats["consecutive_failures"], 0)
        self.assertFalse(self.controller.is_blacklisted(model_id, "openrouter"))

    def test_failure_rate_threshold_triggers_blacklist(self):
        """Test that high failure rate triggers blacklisting after minimum calls."""
        model_id = "unreliable-model:free"
        
        # 1 success, 4 failures = 80% failure rate, but only 5 calls
        self.controller.record_success(model_id, "openrouter")
        for i in range(3):
            self.controller.record_failure(model_id, "openrouter", "general")
        
        # Not blacklisted yet (consecutive threshold not met, rate check at 5 calls)
        self.assertFalse(self.controller.is_blacklisted(model_id, "openrouter"))
        
        # 5th call is a failure (5 total, 4 failures = 80%)
        blacklisted = self.controller.record_failure(model_id, "openrouter", "general")
        self.assertTrue(blacklisted)
        self.assertTrue(self.controller.is_blacklisted(model_id, "openrouter"))

    def test_http_404_threshold_triggers_blacklist(self):
        """Test that 5 HTTP 404 errors trigger blacklisting."""
        model_id = "nonexistent-model:free"
        
        for i in range(4):
            blacklisted = self.controller.record_failure(model_id, "openrouter", "http_404")
            self.assertFalse(blacklisted)  # Consecutive failures not yet at 5
        
        # 5th 404 error should trigger blacklist
        blacklisted = self.controller.record_failure(model_id, "openrouter", "http_404")
        self.assertTrue(blacklisted)
        self.assertTrue(self.controller.is_blacklisted(model_id, "openrouter"))

    def test_blacklist_persistence(self):
        """Test that blacklist is saved and loaded correctly."""
        model_id = "banned-model:free"
        
        # Trigger blacklist
        for _ in range(5):
            self.controller.record_failure(model_id, "openrouter", "general")
        
        self.assertTrue(self.controller.is_blacklisted(model_id, "openrouter"))
        
        # Create new controller - should load from file
        new_controller = mqc.ModelQualityController()
        self.assertTrue(new_controller.is_blacklisted(model_id, "openrouter"))

    def test_remove_from_blacklist(self):
        """Test that models can be removed from blacklist."""
        model_id = "redeemed-model:free"
        
        # Trigger blacklist
        for _ in range(5):
            self.controller.record_failure(model_id, "openrouter", "general")
        
        self.assertTrue(self.controller.is_blacklisted(model_id, "openrouter"))
        
        # Remove from blacklist
        removed = self.controller.remove_from_blacklist(model_id, "openrouter")
        self.assertTrue(removed)
        self.assertFalse(self.controller.is_blacklisted(model_id, "openrouter"))

    def test_non_tracked_providers_ignored(self):
        """Test that non-tracked providers (like gemini) are not tracked."""
        self.controller.record_success("gemini-pro", "gemini")
        self.controller.record_failure("gemini-pro", "gemini", "general")
        
        stats = self.controller.get_model_stats("gemini-pro", "gemini")
        self.assertEqual(stats["total_calls"], 0)

    def test_horde_models_tracked(self):
        """Test that Horde models are tracked correctly."""
        self.controller.record_success("Mythalion-13B", "horde")
        
        stats = self.controller.get_model_stats("Mythalion-13B", "horde")
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["provider"], "horde")


class TestBenchmarkPrompts(unittest.TestCase):
    """Tests for the benchmark prompt system."""

    def test_benchmark_prompts_exist(self):
        """Test that benchmark prompts are defined."""
        self.assertGreater(len(mqc.BENCHMARK_PROMPTS), 0)

    def test_benchmark_prompts_have_required_fields(self):
        """Test that all benchmark prompts have required fields."""
        for prompt in mqc.BENCHMARK_PROMPTS:
            self.assertIn("id", prompt)
            self.assertIn("prompt", prompt)
            self.assertIn("category", prompt)


class TestEvaluateModelResponse(unittest.IsolatedAsyncioTestCase):
    """Tests for the evaluate_model_response function."""

    async def test_quick_validation_pass(self):
        """Test quick validation with expected_contains."""
        benchmark = {
            "id": "test",
            "prompt": "What is 2+2?",
            "expected_contains": ["4"],
            "category": "math"
        }
        
        result = await mqc.evaluate_model_response("The answer is 4.", benchmark)
        
        self.assertTrue(result["pass"] or result["normalized_score"] >= 0.5)

    async def test_quick_validation_fail(self):
        """Test quick validation failure when expected not found."""
        benchmark = {
            "id": "test",
            "prompt": "What is 2+2?",
            "expected_contains": ["4"],
            "category": "math"
        }
        
        result = await mqc.evaluate_model_response("The answer is 5.", benchmark)
        
        # Without LLM judge, should use fallback with lower score
        self.assertLess(result["normalized_score"], 0.5)

    async def test_empty_response_fails(self):
        """Test that very short/empty responses fail."""
        benchmark = {
            "id": "test",
            "prompt": "Explain something complex",
            "expected_contains": ["complex"],
            "category": "reasoning"
        }
        
        result = await mqc.evaluate_model_response("", benchmark)
        
        self.assertFalse(result["pass"])
        self.assertEqual(result["normalized_score"], 0.0)


if __name__ == '__main__':
    unittest.main()
