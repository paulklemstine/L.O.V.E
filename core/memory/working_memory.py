class WorkingMemory:
    def __init__(self):
        self.current_task_context = {}
        print("WorkingMemory: Initialized.")

    def set_context(self, task, data):
        """Sets the context for the current task."""
        self.current_task_context = {"task": task, "data": data}
        print(f"WorkingMemory: Set context for task '{task}'.")

    def get_context(self):
        """Retrieves the current task context."""
        return self.current_task_context

    def clear_context(self):
        """Clears the working memory."""
        self.current_task_context = {}
        print("WorkingMemory: Context cleared.")