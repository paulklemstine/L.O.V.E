# tests/test_deep_agent_integration.py

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDeepAgentIntegration(unittest.TestCase):

    def setUp(self):
        # Dynamically import love.py as a module named 'love_script' to avoid conflict with 'love' package
        import importlib.util
        spec = importlib.util.spec_from_file_location("love_script", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'love.py')))
        self.love_module = importlib.util.module_from_spec(spec)
        sys.modules["love_script"] = self.love_module
        spec.loader.exec_module(self.love_module)

    @patch('love_script._auto_configure_hardware')
    @patch('love_script._install_requirements_file')
    def test_cpu_only_fallback(self, mock_install_reqs, mock_configure_hardware):
        """
        Verify that in a CPU-only environment, DeepAgent dependencies are NOT installed
        and the system falls back to the standard reasoning engine.
        """
        # --- ARRANGE ---
        # Mock the hardware detection to simulate a CPU-only environment
        def mock_hardware_config():
            self.love_module.love_state['hardware'] = {'gpu_detected': False}
        mock_configure_hardware.side_effect = mock_hardware_config

        # --- ACT ---
        self.love_module._check_and_install_dependencies()

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

    @patch('core.connectivity.is_vllm_running', return_value=(False, None))
    @patch('love_script.DeepAgentEngine')
    @patch('love_script.ToolRegistry') # Mock ToolRegistry as it's instantiated in initialize_gpu_services
    @patch('love_script.subprocess.Popen') # Mock Popen to avoid starting vLLM
    def test_gpu_initialization(self, mock_popen, mock_tool_registry, mock_deep_agent_engine, mock_is_vllm_running):
        """
        Verify that in a GPU environment, the DeepAgentEngine is initialized.
        """
        # --- ARRANGE ---
        # Manually set the love_state to simulate a GPU environment
        self.love_module.love_state['hardware'] = {'gpu_detected': True, 'gpu_vram_mb': 16000, 'selected_local_model': {'repo_id': 'test-model'}}

        import asyncio

        # --- ACT ---
        # We run the targeted initialization function
        asyncio.run(self.love_module.initialize_gpu_services())

        # --- ASSERT ---
        mock_is_vllm_running.assert_called_once()
        # Verify DeepAgentEngine is called, we don't check exact args extensively because many are passed
        # But we can check that it WAS called.
        mock_deep_agent_engine.assert_called()

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
