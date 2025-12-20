
import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing core
sys.modules["rich"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rich.panel"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()
sys.modules["rich.text"] = MagicMock()
sys.modules["bbs"] = MagicMock()
sys.modules["display"] = MagicMock()
sys.modules["ipfs"] = MagicMock()
sys.modules["ui_utils"] = MagicMock()
sys.modules["core.logging"] = MagicMock()
sys.modules["core.capabilities"] = MagicMock()
sys.modules["core.token_utils"] = MagicMock()

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from unittest.mock import patch
# Now import should safe
from core.llm_api import run_llm, EmergencyMemoryFold

class TestTokenBudget(unittest.TestCase):
    @patch('core.llm_api.get_token_count')
    @patch('core.llm_api.rank_models')
    @patch('core.llm_api.MODEL_CONTEXT_SIZES', {'test-model-small': 100, 'test-model-large': 1000})
    @patch('core.llm_api.LLM_AVAILABILITY', {'test-model-small': 0, 'test-model-large': 0})
    @patch('core.llm_api.PROVIDER_AVAILABILITY', {'test': 0})
    @patch('core.llm_api.MODEL_STATS')
    @patch('core.llm_api._pin_to_ipfs_async')
    def test_token_budget_monitor(self, mock_ipfs, mock_stats, mock_rank, mock_count):
        
        # Setup
        mock_stats.__getitem__.side_effect = lambda key: {"provider": "test"}
        mock_count.return_value = 500 # Request size is 500
        mock_ipfs.return_value = "cid"
        
        # Case 1: models = [test-model-small] -> Limit 100. 500+2000 > 100. Fail.
        mock_rank.return_value = ['test-model-small']
        
        async def run_test_fail():
            try:
                # We need to mock _models_initialized to avoid network call
                with patch('core.llm_api._models_initialized', True):
                    await run_llm(prompt_text="test", purpose="test")
                return "No Exception"
            except EmergencyMemoryFold:
                return "Caught Exception"
            except Exception as e:
                return f"Other Exception: {e}"
                
        res = asyncio.run(run_test_fail())
        self.assertEqual(res, "Caught Exception")
        
if __name__ == '__main__':
    unittest.main()
