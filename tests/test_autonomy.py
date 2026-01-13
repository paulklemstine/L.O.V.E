import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import importlib.util
import sys

# Load love.py as a module to avoid name collision with the 'love' package
spec = importlib.util.spec_from_file_location("love", "love.py")
love_script = importlib.util.module_from_spec(spec)
sys.modules['love_script'] = love_script
spec.loader.exec_module(love_script)

from core.task import Task
_prioritize_and_select_task = love_script._prioritize_and_select_task
shared_state = love_script.shared_state

class TestAutonomy(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Reset shared_state for each test
        shared_state.love_state = {
            'critical_error_queue': [],
            'proactive_leads': [],
            'autopilot_goal': 'Test Goal'
        }
        shared_state.love_task_manager = MagicMock()
        shared_state.love_task_manager.is_duplicate_task = AsyncMock(return_value=False)

    def test_task_creation(self):
        """Tests the basic creation of a Task object."""
        task = Task(description="Test task", source="Test")
        self.assertEqual(task.description, "Test task")
        self.assertEqual(task.source, "Test")
        self.assertEqual(task.status, "pending")
        self.assertIsNotNone(task.id)

    def test_task_creation_failures(self):
        """Tests that Task creation fails with invalid input."""
        with self.assertRaises(ValueError):
            Task(description="", source="Test")
        with self.assertRaises(ValueError):
            Task(description="Test", source="")

    @patch('love_script.run_llm', new_callable=AsyncMock)
    async def test_prioritization_with_llm(self, mock_run_llm):
        """Tests the task prioritization and selection logic with a mocked LLM."""
        # Setup mock LLM response
        mock_run_llm.return_value = {
            "result": "task_id_1: 0.9\ntask_id_2: 0.5\ntask_id_3: 0.1"
        }

        # Use mock_uuid to control the generated task IDs
        with patch('uuid.uuid4', side_effect=["task_id_1", "task_id_2", "task_id_3"]):
            shared_state.love_state['critical_error_queue'].append({'status': 'new', 'message': 'Test error'})
            shared_state.love_state['proactive_leads'].append({'status': 'new', 'type': 'ip', 'value': '127.0.0.1', 'source': 'test'})

            selected_task = await _prioritize_and_select_task()

            self.assertIsNotNone(selected_task)
            self.assertEqual(selected_task.id, "task_id_1")
            self.assertEqual(selected_task.priority_score, 0.9)

    @patch('love_script.run_llm', new_callable=AsyncMock)
    async def test_prioritization_llm_failure_fallback(self, mock_run_llm):
        """Tests the fallback logic when the LLM fails to return valid scores."""
        mock_run_llm.side_effect = Exception("LLM API is down")

        shared_state.love_state['critical_error_queue'].append({'status': 'new', 'message': 'High priority error'})
        shared_state.love_state['proactive_leads'].append({'status': 'new', 'type': 'ip', 'value': '127.0.0.1', 'source': 'test'})

        selected_task = await _prioritize_and_select_task()

        self.assertIsNotNone(selected_task)
        # Fallback logic should prioritize Error Correction
        self.assertEqual(selected_task.source, "Error Correction")

if __name__ == '__main__':
    unittest.main()
