from abc import ABC, abstractmethod
from typing import Dict

class SpecialistAgent(ABC):
    """
    Abstract base class for a Specialist Agent.

    Specialist agents are designed to perform specific, well-defined tasks
    as part of a larger workflow orchestrated by a Supervisor agent.
    """

    @abstractmethod
    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes the specialist's specific task.

        Args:
            task_details: A dictionary containing all the necessary information
                          and context for the agent to perform its task.

        Returns:
            A dictionary containing the results of the task, which must include
            a 'status' key ('success' or 'failure') and a 'result' key.
        """
        pass
