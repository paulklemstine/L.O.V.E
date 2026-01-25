import unittest
import os
import json
from config_manager import ConfigurationManager

class TestConfigurationManager(unittest.TestCase):

    def setUp(self):
        self.test_file = 'test_config.json'
        # Ensure the file does not exist before each test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        # Clean up the test file after each test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_initialization_no_file(self):
        """Test initialization when the config file does not exist."""
        cm = ConfigurationManager(self.test_file)
        self.assertEqual(cm.static_rules, {})
        self.assertEqual(cm.active_goals, {})
        self.assertEqual(cm.work_queue, [])
        self.assertFalse(os.path.exists(self.test_file))

    def test_save_and_load_state(self):
        """Test saving the state to a file and loading it back."""
        cm_save = ConfigurationManager(self.test_file)
        cm_save.static_rules = {'rule1': 'test'}
        cm_save.update_goal('goal1', 'achieve test')
        cm_save.add_task({'task_id': 'test_01', 'status': 'testing'})
        cm_save.save_state()

        self.assertTrue(os.path.exists(self.test_file))

        cm_load = ConfigurationManager(self.test_file)
        self.assertEqual(cm_load.static_rules, {'rule1': 'test'})
        self.assertEqual(cm_load.active_goals, {'goal1': 'achieve test'})
        self.assertEqual(cm_load.work_queue, [{'task_id': 'test_01', 'status': 'testing'}])

    def test_update_goal(self):
        """Test the update_goal method."""
        cm = ConfigurationManager(self.test_file)
        cm.update_goal('primary', 'win')
        self.assertEqual(cm.active_goals, {'primary': 'win'})
        cm.update_goal('primary', 'win big')
        self.assertEqual(cm.active_goals, {'primary': 'win big'})

    def test_add_task(self):
        """Test the add_task method."""
        cm = ConfigurationManager(self.test_file)
        task1 = {'id': 1, 'desc': 'first task'}
        cm.add_task(task1)
        self.assertEqual(cm.work_queue, [task1])
        task2 = {'id': 2, 'desc': 'second task'}
        cm.add_task(task2)
        self.assertEqual(cm.work_queue, [task1, task2])

    def test_initialization_with_existing_file(self):
        """Test initialization when the config file already exists."""
        data = {
            "static_rules": {"rule": "loaded"},
            "active_goals": {"goal": "loaded"},
            "work_queue": [{"task": "loaded"}]
        }
        with open(self.test_file, 'w') as f:
            json.dump(data, f)

        cm = ConfigurationManager(self.test_file)
        self.assertEqual(cm.static_rules, {"rule": "loaded"})
        self.assertEqual(cm.active_goals, {"goal": "loaded"})
        self.assertEqual(cm.work_queue, [{"task": "loaded"}])

if __name__ == '__main__':
    unittest.main()
