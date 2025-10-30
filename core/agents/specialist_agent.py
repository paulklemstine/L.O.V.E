from abc import ABC, abstractmethod
from typing import Dict

class SpecialistAgent(ABC):
    """
    Abstract base class for all specialist agents. It ensures that each
    specialist agent has a consistent interface for execution.
    """
    @abstractmethod
    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes a specific task for the specialist agent.

        Args:
            task_details: A dictionary containing the parameters for the task.

        Returns:
            A dictionary containing the result of the task, including a
            'status' ('success' or 'failure') and a 'result' string.
        """
        pass
