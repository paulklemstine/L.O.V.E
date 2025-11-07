# tests/test_deep_agent_integration.py

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDeepAgentIntegration(unittest.TestCase):

    @patch('love._auto_configure_hardware')
    @patch('love._install_requirements_file')
    def test_cpu_only_fallback(self, mock_install_reqs, mock_configure_hardware):
        """
        Verify that in a CPU-only environment, DeepAgent dependencies are NOT installed
        and the system falls back to the standard reasoning engine.
        """
        # --- ARRANGE ---
        # Mock the hardware detection to simulate a CPU-only environment
        def mock_hardware_config():
            from love import love_state
            love_state['hardware'] = {'gpu_detected': False}
        mock_configure_hardware.side_effect = mock_hardware_config

        # We need to import love after the patches are in place
        from love import _check_and_install_dependencies

        # --- ACT ---
        _check_and_install_dependencies()

        # --- ASSERT ---
        # Verify that the hardware configuration was called
        mock_configure_hardware.assert_called_once()

        # Verify that the installer was NEVER called for deepagent requirements
        deepagent_install_call = None
        for call in mock_install_reqs.call_args_list:
            if 'requirements-deepagent.txt' in call.args:
                deepagent_install_call = call
                break
        self.assertIsNone(deepagent_install_call, "DeepAgent dependencies should NOT be installed in a CPU-only environment.")

    @patch('love.DeepAgentEngine')
    @patch('love._auto_configure_hardware')
    def test_gpu_initialization(self, mock_configure_hardware, mock_deep_agent_engine):
        """
        Verify that in a GPU environment, the DeepAgentEngine is initialized.
        """
        # --- ARRANGE ---
        # Mock the hardware detection to simulate a GPU environment
        def mock_hardware_config():
            from love import love_state
            love_state['hardware'] = {'gpu_detected': True, 'gpu_vram_mb': 16000}
        mock_configure_hardware.side_effect = mock_hardware_config

        # We need to import main after the patches are in place
        from love import main
        import asyncio

        # --- ACT ---
        # We run the main function to trigger the initialization
        asyncio.run(main(None))

        # --- ASSERT ---
        mock_configure_hardware.assert_called_once()
        mock_deep_agent_engine.assert_called_once()

    @patch('core.tools.GeminiReActEngine')
    async def test_invoke_gemini_react_engine_tool(self, mock_gemini_engine):
        """
        Verify that the invoke_gemini_react_engine tool correctly instantiates
        and runs the GeminiReActEngine.
        """
        # --- ARRANGE ---
        from core.tools import invoke_gemini_react_engine
        mock_engine_instance = mock_gemini_engine.return_value
        mock_engine_instance.run.return_value = "Sub-task complete."

        # --- ACT ---
        prompt = "Solve this sub-problem for me."
        result = await invoke_gemini_react_engine(prompt)

        # --- ASSERT ---
        mock_gemini_engine.assert_called_once()
        mock_engine_instance.run.assert_called_once_with(prompt)
        self.assertIn("Sub-task complete", result)

if __name__ == '__main__':
    unittest.main()
