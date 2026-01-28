import sys
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.service_manager import ServiceManager

class TestColabStartup(unittest.TestCase):
    def setUp(self):
        self.held_output = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.held_output

    def tearDown(self):
        sys.stdout = self.original_stdout

    def test_service_manager_in_colab(self):
        # Simulate Colab
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            sm = ServiceManager(project_root)
            result = sm.start_vllm()
            
            self.assertTrue(result, "start_vllm should return True in Colab")
            
            output = self.held_output.getvalue()
            self.assertIn("Running in Google Colab", output)
            self.assertIn("Skipping local vLLM startup", output)
            self.assertNotIn("Starting vLLM server", output) # Should NOT attempt to start

    def test_is_colab_helper(self):
        sm = ServiceManager(project_root)
        
        # Test Not Colab (current environment)
        # Ensure google.colab is NOT in sys.modules for this test if it wasn't already
        with patch.dict('sys.modules'):
            if 'google.colab' in sys.modules:
                del sys.modules['google.colab']
            self.assertFalse(sm.is_colab())

        # Test With Colab
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            self.assertTrue(sm.is_colab())

if __name__ == '__main__':
    unittest.main()
