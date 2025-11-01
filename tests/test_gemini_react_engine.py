import unittest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.agents.orchestrator import Orchestrator

class TestOrchestratorEndToEnd(unittest.IsolatedAsyncioTestCase):
    @patch("core.agents.orchestrator.run_llm", new_callable=AsyncMock)
    async def test_end_to_end_integration(self, mock_run_llm):
        # Mock the planner LLM call to return a pre-defined plan
        mock_run_llm.return_value = {
            "result": json.dumps({
                "plan": [
                    {"step": 1, "task": "Find the current temperature in New York City.", "agent": "WebAutomationAgent", "dependencies": []},
                    {"step": 2, "task": "Generate an image of a snowman based on the weather.", "agent": "ImageGenerationAgent", "dependencies": [1]}
                ]
            })
        }

        # Mock the MemoryManager
        mock_memory_manager = MagicMock()
        mock_memory_manager.add_episode = AsyncMock()

        # Create an instance of the Orchestrator
        orchestrator = Orchestrator(mock_memory_manager)

        # Mock the _create_specialist method to return mock specialists
        mock_web_automation_agent = MagicMock()
        mock_web_automation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'The temperature is 32 degrees.'})

        mock_image_generation_agent = MagicMock()
        mock_image_generation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'Image of a snowman generated.'})

        def create_specialist_side_effect(agent_name, **kwargs):
            if agent_name == "WebAutomationAgent":
                return mock_web_automation_agent
            elif agent_name == "ImageGenerationAgent":
                return mock_image_generation_agent
            return MagicMock()

        orchestrator._create_specialist = MagicMock(side_effect=create_specialist_side_effect)

        # Execute the goal
        goal = "Find out the current temperature in New York City and then generate an image of a snowman."
        await orchestrator.execute_goal(goal)

        # Assert that the correct specialists were called and that context was passed correctly
        mock_web_automation_agent.execute_task.assert_called_once_with(
            {'step': 1, 'task': 'Find the current temperature in New York City.', 'agent': 'WebAutomationAgent', 'dependencies': []},
            {}
        )
        mock_image_generation_agent.execute_task.assert_called_once_with(
            {'step': 2, 'task': 'Generate an image of a snowman based on the weather.', 'agent': 'ImageGenerationAgent', 'dependencies': [1]},
            {'1': 'The temperature is 32 degrees.'}
        )

if __name__ == "__main__":
    unittest.main()
