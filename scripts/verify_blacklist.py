"""
verify_blacklist.py

Verifies:
1. Blacklist loading/saving.
2. Blacklist checking during start_vllm.
3. Adding to blacklist on failure.
"""

import sys
import os
import unittest
import json
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules["huggingface_hub"] = MagicMock()

from core.service_manager import ServiceManager

class TestBlacklist(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = Path("test_blacklist_env")
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "logs").mkdir(exist_ok=True)
        self.bl_file = self.test_dir / ".vllm_blacklist.json"
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_blacklist_io(self):
        """Test load/save/check."""
        sm = ServiceManager(self.test_dir)
        
        # Should be empty initially
        self.assertEqual(sm._load_blacklist(), [])
        
        # Add item
        sm._add_to_blacklist("ModelA", 8192)
        sm._add_to_blacklist("ModelB", None)
        
        # Verify file content
        with open(self.bl_file, 'r') as f:
            data = json.load(f)["blacklist"]
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0], {"model": "ModelA", "max_len": 8192})
            self.assertEqual(data[1], {"model": "ModelB", "max_len": None})
            
        # Verify check
        self.assertTrue(sm._is_blacklisted("ModelA", 8192))
        self.assertTrue(sm._is_blacklisted("ModelB", None))
        self.assertFalse(sm._is_blacklisted("ModelA", 4096))
        
    @patch('core.service_manager.ServiceManager._launch_process')
    @patch('core.model_selector.ModelSelector.select_best_models') # Skip actual selection
    @patch('core.service_manager.ServiceManager.get_total_vram_mb')
    def test_blacklist_flow(self, mock_vram, mock_select, mock_launch):
        """Test full flow: Failure -> Blacklist -> Skip next time."""
        
        sm = ServiceManager(self.test_dir)
        sm.ensure_vllm_setup = MagicMock(return_value=True)
        sm.is_vllm_healthy = MagicMock(return_value=False)
        sm.load_config = MagicMock(return_value={})
        sm.save_config = MagicMock()
        mock_vram.return_value = 24000
        
        # Candidate chain
        # 1. candidate_queue = [{"name": "FailingModel", "max_len": None}]
        
        # Mock selector to return FailingModel
        mock_model = MagicMock()
        mock_model.name = "FailingModel"
        mock_model.repo_id = "org/FailingModel"
        mock_select.return_value = [mock_model]
        
        # First Run: Launch fails
        mock_launch.return_value = False
        
        print("\n--- Running Attempt 1 (Expect Failure & Blacklist) ---")
        sm.start_vllm()
        
        # Verify it was added to blacklist
        self.assertTrue(sm._is_blacklisted("org/FailingModel", None))
        self.assertTrue(sm._is_blacklisted("org/FailingModel", 16384)) # Logic tries reduced contexts too
        
        print(f"Blacklist content after run 1: {sm._load_blacklist()}")
        
        # Reset launch mock to track calls for second run
        mock_launch.reset_mock()
        mock_launch.return_value = True 
        
        # Second Run
        mock_model2 = MagicMock()
        mock_model2.name = "WorkingModel"
        mock_model2.repo_id = "org/WorkingModel"
        mock_select.return_value = [mock_model, mock_model2]
        
        print("\n--- Running Attempt 2 ---")
        sm.start_vllm()
        
        calls = [c[0][0] for c in mock_launch.call_args_list]
        print(f"Launch Calls in Run 2: {calls}")
        
        self.assertNotIn("org/FailingModel", calls)
        self.assertIn("org/WorkingModel", calls)

if __name__ == '__main__':
    unittest.main()
