import unittest
import os
import json
import asyncio
from core.evolution_analyzer import determine_evolution_goal, _find_codebase_hotspot

class TestEvolutionAnalyzer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Clean up any residual files from previous failed runs
        if os.path.exists("love.log"):
            os.remove("love.log")
        if os.path.exists("hotspot.py"):
            os.remove("hotspot.py")

    def tearDown(self):
        # Clean up files created during tests
        if os.path.exists("love.log"):
            os.remove("love.log")
        if os.path.exists("hotspot.py"):
            os.remove("hotspot.py")

    async def test_priority_1_recurring_critical_errors(self):
        """The highest priority should be fixing recurring critical errors from the state."""
        love_state = {
            "critical_error_queue": [
                {"message": "Critical Error A\\nDetails..."},
                {"message": "Critical Error B\\nDetails..."},
                {"message": "Critical Error A\\nDetails..."},
                {"message": "Critical Error A\\nDetails..."},
            ],
            "autopilot_history": [{"command": "some_command"}] # Add other data to ensure priority is correct
        }
        with open("love.log", "w") as f:
            f.write("ERROR: Some other error\\n")

        goal = await determine_evolution_goal(love_state=love_state, exclude_files=['love.py'])
        self.assertIn("Fix the recurring critical error from state: 'Critical Error A'", goal)

    async def test_priority_2_recurring_log_errors(self):
        """If no critical errors in state, the next priority is recurring errors from logs."""
        love_state = { "critical_error_queue": [] }
        with open("love.log", "w") as f:
            f.write("ERROR: Test error 2\\n" * 4) # 4 times to meet the threshold

        goal = await determine_evolution_goal(love_state=love_state, exclude_files=['love.py'])
        self.assertIn("Fix the recurring error from logs: 'Test error 2'", goal)

    async def test_priority_3_code_hotspots(self):
        """If no errors, the next priority is refactoring a hotspot."""
        love_state = {}
        # No log file, or an empty one
        if os.path.exists("love.log"):
            os.remove("love.log")
        with open("hotspot.py", "w") as f:
            f.write("print('hello')\\n" * 200) # Make it a hotspot

        # We must call the hotspot function directly to ensure it finds our file
        hotspot, issue, score = _find_codebase_hotspot(exclude_files=['love.py'])
        self.assertEqual(hotspot, 'hotspot.py')

        goal = await determine_evolution_goal(love_state=love_state, exclude_files=['love.py'])
        self.assertIn("Refactor 'hotspot.py'", goal)

    async def test_priority_4_command_patterns(self):
        """If no errors or hotspots, the next priority is enhancing a frequent command."""
        love_state = {
            "autopilot_history": [
                {"command": "test_command_1"},
                {"command": "test_command_2"},
                {"command": "test_command_1"},
                {"command": "test_command_1"},
            ]
        }
        # Ensure no other higher-priority conditions are met
        if os.path.exists("love.log"):
            os.remove("love.log")
        if os.path.exists("hotspot.py"):
            os.remove("hotspot.py")

        goal = await determine_evolution_goal(love_state=love_state, exclude_files=['love.py'])
        self.assertIn("Enhance the 'test_command_1' command", goal)

    async def test_fallback_goal(self):
        """If no other conditions are met, it should return the fallback goal."""
        love_state = {}
        if os.path.exists("love.log"):
            os.remove("love.log")
        if os.path.exists("hotspot.py"):
            os.remove("hotspot.py")

        goal = await determine_evolution_goal(love_state=love_state, exclude_files=['love.py'])
        self.assertIn("Conduct a general codebase review", goal)

if __name__ == '__main__':
    unittest.main()
