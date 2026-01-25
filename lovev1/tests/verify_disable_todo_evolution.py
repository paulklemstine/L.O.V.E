
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might cause import errors or side effects
sys.modules['core.llm_api'] = MagicMock()

# We need core.logging to exist but we can mock its methods
mock_logging = MagicMock()
sys.modules['core.logging'] = mock_logging

# CRITICAL: We must ensure 'core' has 'logging' attribute
import core
core.logging = mock_logging

# Now import the module under test
from core.evolution_analyzer import determine_evolution_goal

class TestDisableTodo(unittest.IsolatedAsyncioTestCase):
    async def test_todo_scanning_is_disabled(self):
        print("Starting test_todo_scanning_is_disabled...")
        with patch('core.evolution_analyzer._scan_codebase_todos') as mock_scan:
            # Setup mock to return something if it WAS called
            mock_scan.return_value = [{"file": "test.py", "line": 1, "text": "TODO: This should not be found"}]
            
            # Run the function
            # We need to mock other things that determining goal might call or fail on
            with patch('core.evolution_analyzer._analyze_recent_logs', return_value=[]), \
                 patch('core.evolution_analyzer._analyze_technical_debt', return_value=[]), \
                 patch('core.evolution_analyzer._query_knowledge_base', return_value=[]), \
                 patch('core.evolution_analyzer._analyze_system_state', return_value=[]), \
                 patch('core.evolution_analyzer._synthesize_goal_with_llm', return_value="Evolution Goal"):
                 
                print("Calling determine_evolution_goal...")
                await determine_evolution_goal()
                print("Returned from determine_evolution_goal.")
                
            # Assert scan was NOT called
            if mock_scan.called:
                print("FAILURE: _scan_codebase_todos WAS called!")
                self.fail("_scan_codebase_todos should not have been called")
            else:
                print("SUCCESS: _scan_codebase_todos was not called.")

if __name__ == "__main__":
    unittest.main()
