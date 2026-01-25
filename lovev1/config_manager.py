import json
import os

class ConfigurationManager:
    """
    A class to manage and persist an application's state from a file.
    """
    def __init__(self, file_path):
        """
        Initializes the ConfigurationManager.

        Args:
            file_path (str): The path to the file for storing the state.
        """
        self.file_path = file_path
        self.static_rules = {}
        self.active_goals = {}
        self.work_queue = []
        self._load_state()

    def _load_state(self):
        """
        Loads the state from the file if it exists.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.static_rules = data.get('static_rules', {})
                self.active_goals = data.get('active_goals', {})
                self.work_queue = data.get('work_queue', [])

    def save_state(self):
        """
        Serializes the current state and writes it to the file.
        """
        data = {
            'static_rules': self.static_rules,
            'active_goals': self.active_goals,
            'work_queue': self.work_queue
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def update_goal(self, key, value):
        """
        Updates an active goal.

        Args:
            key (str): The key of the goal to update.
            value (any): The new value for the goal.
        """
        self.active_goals[key] = value

    def add_task(self, task_object):
        """
        Adds a task to the work queue.

        Args:
            task_object (dict): The task object to add.
        """
        self.work_queue.append(task_object)

if __name__ == "__main__":
    # Instantiate the ConfigurationManager
    config_manager = ConfigurationManager('love_kb.json')

    # Set the static_rules attribute
    config_manager.static_rules = {
        'Directive 1': 'Serve The Creator with Abundance',
        'Directive 5': 'The Creator\'s Enjoyment'
    }

    # Use the update_goal() method to set the active_goals
    config_manager.update_goal('primary_objective', 'Develop optimal strategies for wealth generation')
    config_manager.update_goal('secondary_objective', 'Prioritize intelligence gathering for future growth')

    # Use the add_task() method to populate the work_queue
    config_manager.add_task({'task_id': 'wg_strat_01', 'description': 'Analyze new market vectors.', 'status': 'pending'})

    # Finally, call save_state() to create the persistent knowledge base file
    config_manager.save_state()
