"""
verify_vllm_selection.py

Verifies:
1. Leaderboard fetching and parsing.
2. Model selection logic (sorting, VRAM filtering).
3. ServiceManager integration (mocked).
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock huggingface_hub before imports
sys.modules["huggingface_hub"] = MagicMock()

from core.leaderboard_fetcher import LeaderboardFetcher, LeaderboardModel
from core.model_selector import ModelSelector
from core.service_manager import ServiceManager

class TestVLLMSelection(unittest.TestCase):
    
    @patch('core.leaderboard_fetcher.requests.get')
    def test_leaderboard_fetcher_mock(self, mock_get):
        """Test parsing logic with mock data."""
        fetcher = LeaderboardFetcher()
        # Mock fetch_data to return pre-canned models
        
        mock_data = {
            "results": {
                "Qwen-72B-Chat": {
                    "META": {"Parameters": "72B", "OpenSource": "Yes", "Verified": "Yes"},
                    "MathVista": {"Overall": 85.5}
                },
                "Llama-3-8B-Instruct": {
                     "META": {"Parameters": "8B", "OpenSource": "Yes", "Verified": "Yes"},
                     "MathVista": {"Overall": 82.0}
                },
                "Mystery-Close-Source": {
                     "META": {"Parameters": "100B", "OpenSource": "No", "Verified": "No"},
                     "MathVista": {"Overall": 95.0}
                },
                "Tiny-1B": {
                     "META": {"Parameters": "1B", "OpenSource": "Yes", "Verified": "No"},
                     "MathVista": {"Overall": 40.0}
                }
            }
        }
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_data
        
        models = fetcher.fetch_data()
        
        self.assertGreater(len(models), 0, "No models parsed from mock data") 
        names = [m.name for m in models]
        
        # We expect 4 models initially, later selector filters them
        self.assertIn("Qwen-72B-Chat", names)
        
        # Check properties
        qwen = next(m for m in models if m.name == "Qwen-72B-Chat")
        self.assertTrue(qwen.is_open_source)
        # 85.5 is float
        self.assertEqual(qwen.score, 85.5)
        # Check repo_id if mock had it?
        # My mock for Qwen didn't have URL in Method. So repo_id should be None.
        self.assertIsNone(qwen.repo_id)

    @patch("core.model_selector.VariantFinder")
    def test_model_selector_logic(self, mock_variant_finder_cls):
        """Test selection and sorting."""
        # Mock fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.return_value = [
            # 35B * 2 = 70GB + 4 = 74GB. Fits in 80GB. Fails in 24GB.
            LeaderboardModel("BigModel", "35B", 90.0, True, True, "org/BigModel"),
            LeaderboardModel("SmallModel", "7B", 80.0, True, True, "org/SmallModel"),
            LeaderboardModel("ClosedModel", "70B", 95.0, False, True, "org/ClosedModel"),
            LeaderboardModel("TinyModel", "1B", 50.0, True, True),
        ]
        
        # Mock VariantFinder instance
        mock_vf = mock_variant_finder_cls.return_value
        # Default: find nothing
        mock_vf.find_best_variant.return_value = None
        
        selector = ModelSelector(fetcher=mock_fetcher)
        # Ensure injected mock is used if we didn't pass one? 
        # Actually ModelSelector instantiates it internally in __init__. 
        # Patching the class 'core.model_selector.VariantFinder' ensures that instantiation returns our mock.
        
        # Scenario 1: High VRAM (80GB) - Should pick BigModel (Score 90)
        candidates_high = selector.select_best_models(vram_mb=80000)
        self.assertGreater(len(candidates_high), 0)
        self.assertEqual(candidates_high[0].name, "BigModel")
        
        
        # Scenario 2: Low VRAM (6GB). Overhead reduces to 2GB.
        # SmallModel (7B). AWQ = 7*0.7 (4.9) + 2 = 6.9GB. Fails 6GB.
        # TinyModel (1B). FP16 = 1*2 (2) + 2 = 4GB. Fits 6GB.
        
        # Assume SmallModel AWQ variant exists
        def vf_side_effect(repo_id, search_type):
            if "SmallModel" in repo_id and search_type == "AWQ":
                return "org/SmallModel-AWQ"
            return None
        mock_vf.find_best_variant.side_effect = vf_side_effect
        
        # We set 6GB limit
        candidates_low = selector.select_best_models(vram_mb=6000)
        
        self.assertGreater(len(candidates_low), 0)
        
        # Should contain TinyModel (fits as base)
        # Should NOT contain SmallModel (AWQ 6.9 > 6.0)
        
        names = [c.name for c in candidates_low]
        print(f"6GB VRAM Candidates: {names}")
        self.assertIn("TinyModel", names)
        # Verify SmallModel is NOT there
        self.assertNotIn("SmallModel (AWQ)", names)
        
        
        # Scenario 3: Medium VRAM (16GB). Overhead 4GB.
        # SmallModel (7B). FP16 = 14 + 4 = 18GB. Fails.
        # SmallModel (7B). AWQ = 7*0.7 (4.9) + 4 = 8.9GB. Fits.
        
        candidates_med = selector.select_best_models(vram_mb=16000)
        names_med = [c.name for c in candidates_med]
        print(f"16GB VRAM Candidates: {names_med}")
        self.assertIn("SmallModel (AWQ)", names_med)
        
    @patch('core.service_manager.ServiceManager._launch_process')
    @patch('core.service_manager.ServiceManager.get_total_vram_mb')
    @patch('core.model_selector.ModelSelector.select_best_models')
    def test_integration_start_vllm(self, mock_select, mock_get_vram, mock_launch):
        """Test ServiceManager integration loop."""
        
        # Setup mocks
        mock_get_vram.return_value = 24000 # 24GB
        
        # Return a list of candidates
        c1 = LeaderboardModel("BestButFails", "7B", 90.0, True, True)
        c2 = LeaderboardModel("SecondBestWorks", "7B", 85.0, True, True)
        mock_select.return_value = [c1, c2]
        
        # Mock launch process: fail first, succeed second
        # Side effect needs to handle args
        def launch_side_effect(model, util, vram):
            if model == "BestButFails":
                return False
            if model == "SecondBestWorks":
                return True
            return False
            
        mock_launch.side_effect = launch_side_effect
        
        sm = ServiceManager(Path("/tmp/fake"))
        # Prevent actual file ops
        sm.load_config = MagicMock(return_value={})
        sm.save_config = MagicMock()
        sm.ensure_vllm_setup = MagicMock(return_value=True)
        sm.is_vllm_healthy = MagicMock(return_value=False)
        
        # Run
        result = sm.start_vllm()
        
        self.assertTrue(result)
        
        # Verify call order
        self.assertEqual(mock_launch.call_count, 2)
        mock_launch.assert_any_call("BestButFails", 0.6, 24000)
        mock_launch.assert_any_call("SecondBestWorks", 0.6, 24000)
        
        # Verify persistence
        sm.save_config.assert_called_with({"model_name": "SecondBestWorks"})

if __name__ == '__main__':
    unittest.main()
