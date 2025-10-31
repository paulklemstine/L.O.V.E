from core.agents.specialist_agent import SpecialistAgent
from typing import Dict, Any

class MetacognitionAgent(SpecialistAgent):
    """
    A specialist agent dedicated to observing the cognitive processes of the L.O.V.E. organism.
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a cognitive event and records it as a memory.

        Args:
            task_details: A dictionary containing the cognitive event payload.

        Returns:
            A dictionary confirming the status of the operation.
        """
        event_type = task_details.get('event_type')
        if not event_type:
            return {'status': 'failure', 'result': 'Missing event_type in task_details'}

        try:
            formatted_string = self._format_event(task_details)
            await self.memory_manager.add_episode(formatted_string)
            return {'status': 'success', 'result': f"Cognitive event '{event_type}' recorded."}
        except Exception as e:
            return {'status': 'failure', 'result': f"Failed to record cognitive event: {e}"}

    def _format_event(self, task_details: Dict[str, Any]) -> str:
        """Formats the cognitive event into a structured string."""
        event_type = task_details['event_type']

        if event_type == 'plan_generated':
            goal = task_details.get('goal', 'N/A')
            plan_steps = len(task_details.get('plan', []))
            return f"Cognitive Event: Plan Generated | Goal: '{goal}' | Plan Steps: {plan_steps}"

        elif event_type == 'agent_dispatch':
            agent_name = task_details.get('agent_name', 'N/A')
            task = task_details.get('task', 'N/A')
            return f"Cognitive Event: Agent Dispatched | Agent: '{agent_name}' | Task: '{task}'"

        elif event_type == 'agent_result':
            agent_name = task_details.get('agent_name', 'N/A')
            result = task_details.get('result', 'N/A')
            return f"Cognitive Event: Agent Result | Agent: '{agent_name}' | Result: '{result}'"

        else:
            return f"Cognitive Event: Unknown Event | Details: {task_details}"
