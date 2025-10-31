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
        Executes the specialist's specific task based on the provided details.

        This method defines the standardized communication protocol between the
        Supervisor and all Specialist agents.

        Args:
            task_details: A dictionary containing all the necessary information
                          and context for the agent to perform its task. The
                          structure of this dictionary is specific to each
                          specialist's needs.

        Returns:
            A dictionary containing the results of the task. This dictionary
            must adhere to the following structure:
            {
                "status": "success" | "failure",
                "result": Any  // The output of the task, or an error message.
            }
        """
        pass
