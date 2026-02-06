"""
verify_model_selection_v2.py

Verifies:
1. Model selection preference: Unquantized > GPTQ.
2. Removal of AWQ/Abliterated support.
3. Correct VRAM estimation for GPTQ.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock huggingface_hub
sys.modules["huggingface_hub"] = MagicMock()

from core.leaderboard_fetcher import LeaderboardModel
from core.model_selector import ModelSelector

class TestGPTQSelection(unittest.TestCase):

    @patch("core.model_selector.VariantFinder")
    def test_gptq_preference(self, mock_variant_finder_cls):
        """Test Unquantized > GPTQ logic."""
        
        # Mock Fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.return_value = [
            # 70B Model: 140GB as FP16. Fails T4 (15GB).
            LeaderboardModel("HugeModel", "70B", 90.0, True, True, "org/HugeModel"),
            # 7B Model: 14GB as FP16. Fits T4 (barely/overhead). 
            # 7B FP16 = 14GB + 4GB overhead = 18GB. Fails T4 (15GB).
            # 7B GPTQ = 5.6GB + 4GB overhead = 9.6GB. Fits T4.
            LeaderboardModel("MediumModel", "7B", 80.0, True, True, "org/MediumModel"),
            # 1B Model: 2GB unquant. Fits T4 easily.
            LeaderboardModel("SmallModel", "1B", 50.0, True, True, "org/SmallModel"),
        ]
        
        # Mock VariantFinder
        mock_vf = mock_variant_finder_cls.return_value
        
        def vf_side_effect(repo_id, search_type):
            if "HugeModel" in repo_id and search_type == "GPTQ":
                return "org/HugeModel-GPTQ-Int4"
            if "MediumModel" in repo_id and search_type == "GPTQ":
                return "org/MediumModel-GPTQ-Int4"
            if search_type == "AWQ":
                return "org/ShouldNotBeCalled"
            return None
            
        mock_vf.find_best_variant.side_effect = vf_side_effect
        
        selector = ModelSelector(fetcher=mock_fetcher)
        
        # Test 1: T4 Environment (15000 MB)
        # Expect:
        # - HugeModel (Unquant) -> Fail
        # - HugeModel (GPTQ) -> 70 * 0.75 = 52.5 + 4 -> 56GB -> Fail
        # - MediumModel (Unquant) -> 14 + 4 = 18GB -> Fail
        # - MediumModel (GPTQ) -> 7 * 0.75 = 5.25 + 4 = 9.25GB -> Fit!
        # - SmallModel (Unquant) -> 1 * 2 = 2 + 4 = 6GB -> Fit!
        
        candidates = selector.select_best_models(vram_mb=15000)
        names = [c.name for c in candidates]
        
        print(f"\nT4 Candidates: {names}")
        
        # Verify MediumModel GPTQ is selected
        self.assertIn("MediumModel (GPTQ)", names)
        # Verify SmallModel (Unquant) is selected
        self.assertIn("SmallModel", names)
        # Verify HugeModel is NOT selected (too big even quantified)
        self.assertNotIn("HugeModel (GPTQ)", names)
        
        # Verify AWQ was NOT moved to candidates even if it existed (we mocked it off in finder anyway)
        # But crucially, make sure we didn't call finder with AWQ
        # Actually ModelSelector logic no longer calls finding for AWQ.
        # We can check mock_vf calls
        
        calls = [c[0] for c in mock_vf.find_best_variant.call_args_list]
        print(f"VariantFinder Calls: {calls}")
        
        # Should call for GPTQ
        # Should NOT call for AWQ
        has_gptq_call = any("GPTQ" in str(args) for args in mock_vf.find_best_variant.call_args_list)
        has_awq_call = any("AWQ" in str(args) for args in mock_vf.find_best_variant.call_args_list)
        
        self.assertTrue(has_gptq_call, "Should search for GPTQ")
        self.assertFalse(has_awq_call, "Should NOT search for AWQ")

if __name__ == '__main__':
    unittest.main()
