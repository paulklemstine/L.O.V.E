# tests/test_select_model.py

import unittest
from unittest.mock import patch
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.deep_agent_engine import DeepAgentEngine

class TestSelectModel(unittest.TestCase):

    @patch('love.love_state', {'hardware': {'gpu_vram_mb': 40000}})
    def test_select_model_high_vram(self):
        """Test model selection for high VRAM."""
        engine = DeepAgentEngine.__new__(DeepAgentEngine)
        model = engine._select_model()
        self.assertEqual(model, "QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ")

    @patch('love.love_state', {'hardware': {'gpu_vram_mb': 20000}})
    def test_select_model_low_vram(self):
        """Test model selection for low VRAM."""
        engine = DeepAgentEngine.__new__(DeepAgentEngine)
        model = engine._select_model()
        self.assertEqual(model, "QuantTrio/GLM-4.5V-AWQ")

if __name__ == '__main__':
    unittest.main()
