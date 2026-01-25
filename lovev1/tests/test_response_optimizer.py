import unittest
import json
import os
import shutil
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from core.agents.response_optimizer import ResponseOptimizer

class TestResponseOptimizer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.test_stats_file = "test_llm_model_stats.json"
        self.stats_data = {
            "slow_model": {
                "total_time_spent": 10.0,
                "successful_calls": 10,
                "total_tokens_generated": 1000
            },
            "chatty_model": {
                "total_time_spent": 1.0,
                "successful_calls": 1,
                "total_tokens_generated": 3000
            }
        }
        with open(self.test_stats_file, 'w') as f:
            json.dump(self.stats_data, f)

        self.tool_registry = MagicMock()
        self.monitoring_manager = MagicMock()
        self.optimizer = ResponseOptimizer(self.tool_registry, self.monitoring_manager)
        self.optimizer.model_stats_file = self.test_stats_file

    def tearDown(self):
        if os.path.exists(self.test_stats_file):
            os.remove(self.test_stats_file)

    async def test_identify_inefficiencies(self):
        stats = self.optimizer._load_model_stats()
        inefficiencies = await self.optimizer._identify_inefficiencies(stats)

        self.assertIn("High latency for slow_model", inefficiencies)
        self.assertIn("High token usage for chatty_model", inefficiencies)

    @patch("core.agents.response_optimizer.run_llm", new_callable=AsyncMock)
    async def test_generate_refactoring_plan(self, mock_run_llm):
        mock_plan = {
            "analysis": "Test analysis",
            "proposed_changes": [
                {
                    "type": "prompt_update",
                    "prompt_key": "test_prompt",
                    "new_prompt_content": "optimized content"
                }
            ]
        }
        mock_run_llm.return_value = {"result": json.dumps(mock_plan)}

        plan = await self.optimizer._generate_refactoring_plan("Test inefficiencies")
        self.assertEqual(plan["analysis"], "Test analysis")
        self.assertEqual(plan["proposed_changes"][0]["prompt_key"], "test_prompt")

    @patch("core.agents.response_optimizer.get_prompt_registry")
    @patch("core.agents.response_optimizer.ResponseOptimizer._validate_improvement", new_callable=AsyncMock)
    async def test_apply_updates_success(self, mock_validate, mock_get_registry):
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_prompt.return_value = "original content"
        mock_registry.update_prompt.return_value = True
        mock_validate.return_value = True

        plan = {
            "proposed_changes": [
                {
                    "type": "prompt_update",
                    "prompt_key": "test_prompt",
                    "new_prompt_content": "optimized content"
                }
            ]
        }

        success = await self.optimizer._apply_updates_with_safeguards(plan)
        self.assertTrue(success)
        mock_registry.update_prompt.assert_called_with("test_prompt", "optimized content")

    @patch("core.agents.response_optimizer.get_prompt_registry")
    @patch("core.agents.response_optimizer.ResponseOptimizer._validate_improvement", new_callable=AsyncMock)
    async def test_apply_updates_rollback(self, mock_validate, mock_get_registry):
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_registry.get_prompt.return_value = "original content"
        mock_registry.update_prompt.return_value = True
        mock_validate.return_value = False # Simulate validation failure

        plan = {
            "proposed_changes": [
                {
                    "type": "prompt_update",
                    "prompt_key": "test_prompt",
                    "new_prompt_content": "optimized content"
                }
            ]
        }

        success = await self.optimizer._apply_updates_with_safeguards(plan)
        self.assertFalse(success)
        # Verify rollback was called
        mock_registry.update_prompt.assert_any_call("test_prompt", "original content")

if __name__ == "__main__":
    unittest.main()
