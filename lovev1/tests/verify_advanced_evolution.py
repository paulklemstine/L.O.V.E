
import unittest
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock core.tracing dependencies BEFORE importing core.tools
mock_langsmith = MagicMock()
sys.modules['langsmith'] = mock_langsmith
sys.modules['langsmith.run_helpers'] = MagicMock() # Ensure submodules exist
sys.modules['langsmith.wrappers'] = MagicMock()

# Mock core.tracing
mock_tracing = MagicMock()
mock_tracing.traceable = lambda x: x
sys.modules['core.tracing'] = mock_tracing

from core.tools import trigger_optimization_pipeline
from core.researcher import explore_structured_data

class TestAdvancedEvolution(unittest.IsolatedAsyncioTestCase):
    
    async def test_optimization_tool_exists_and_runs(self):
        """Test that the tool attempts to run the subprocess."""
        
        # We check if the script file exists first
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "optimize_prompts.py")
        self.assertTrue(os.path.exists(script_path), "optimize_prompts.py should exist")
        
        # We verify the tool function logic by mocking subprocess.run
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Optimization simulation complete."
            
            result = await trigger_optimization_pipeline("test_key", "justification")
            
            self.assertIn("Optimization complete", result)
            mock_run.assert_called_once()
    
    async def test_schema_generation_integration(self):
        """Test that explore_structured_data calls the schema generator."""
        
        # Use patch to mock the subprocess calls (schema gen) and webrequest/llm
        with patch('subprocess.run') as mock_sub, \
             patch('network.perform_webrequest', new_callable=MagicMock) as mock_web, \
             patch('core.llm_api.run_llm', new_callable=MagicMock) as mock_llm:
                 
            # Mock Schema Gen Output
            mock_sub.return_value.returncode = 0
            mock_sub.return_value.stdout = "class ExtractedData(BaseModel): pass"
            
            # Mock Web Request
            async def async_web_ret(*args): return "Simulated web content", None
            mock_web.side_effect = async_web_ret
            
            # Mock LLM Extraction
            async def async_llm_ret(*args, **kwargs): return {"result": '{"foo": "bar"}'}
            mock_llm.side_effect = async_llm_ret
            
            result = await explore_structured_data("test_topic", "test_schema")
            
            self.assertEqual(result, {"foo": "bar"})
            
            # Verify subprocess call for schema gen
            args = mock_sub.call_args[0][0]
            self.assertIn("generate_schema.py", args[1])

if __name__ == '__main__':
    unittest.main()
