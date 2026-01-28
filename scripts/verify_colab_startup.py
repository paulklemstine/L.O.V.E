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

    def test_service_manager_in_colab_default(self):
        # Simulate Colab with default settings (SKIP_VLLM_IN_COLAB = False)
        # It should ATTEMPT to start vLLM (which we'll mock to avoid actual process spawning complexity in this rapid test)
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            sm = ServiceManager(project_root)
            # We mock run logic to prevent actual subproc release, just check flow
            sm.ensure_vllm_setup = MagicMock(return_value=True)
            sm.is_vllm_healthy = MagicMock(return_value=False)
            sm.scripts_dir = MagicMock()
            
            with patch('subprocess.Popen') as mock_popen, \
                 patch('builtins.open', MagicMock()):
                mock_popen.return_value.pid = 1234
                sm.wait_for_vllm = MagicMock(return_value=True)
                sm.get_total_vram_mb = MagicMock(return_value=None)


                result = sm.start_vllm()
                self.assertTrue(result)
                
                # Should NOT have skipped
                output = self.held_output.getvalue()
                self.assertNotIn("Skipping local vLLM startup", output)
                self.assertIn("Starting vLLM server", output)

    def test_service_manager_in_colab_skip_enabled(self):
        # Simulate Colab with SKIP_VLLM_IN_COLAB = True
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            sm = ServiceManager(project_root)
            sm.SKIP_VLLM_IN_COLAB = True
            
            result = sm.start_vllm()
            
            self.assertTrue(result)
            output = self.held_output.getvalue()
            self.assertIn("Skipping local vLLM startup", output)

    def test_service_manager_with_system_vllm(self):
        # Test that proper vLLM install in system python skips setup and uses system mode
        # Mocking setup such that import vllm works
        with patch.dict('sys.modules', {'google.colab': MagicMock(), 'vllm': MagicMock()}):
            sm = ServiceManager(project_root)
            sm.ensure_vllm_setup()
            
            self.assertTrue(getattr(sm, 'use_system_vllm', False))
            
            # Now test correct command generation in start_vllm
            with patch('subprocess.Popen') as mock_popen, \
                 patch('builtins.open', MagicMock()):
                
                sm.is_vllm_healthy = MagicMock(return_value=False)
                mock_popen.return_value.pid = 9999
                sm.wait_for_vllm = MagicMock(return_value=True)
                sm.get_total_vram_mb = MagicMock(return_value=None)
                
                res = sm.start_vllm()
                
                self.assertTrue(res)
                
                # Verify command arg (should NOT have --venv)
                args, kwargs = mock_popen.call_args
                cmd_list = args[0]
                self.assertNotIn("--venv", cmd_list)
                self.assertIn("bash", cmd_list)
                # Should verify start_vllm.sh is in there
                self.assertTrue(any("start_vllm.sh" in str(arg) for arg in cmd_list))

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
