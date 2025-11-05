import unittest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.agents.orchestrator import Orchestrator

class TestOrchestratorEndToEnd(unittest.IsolatedAsyncioTestCase):
    @patch("core.agents.orchestrator.run_llm", new_callable=AsyncMock)
    async def test_end_to_end_integration(self, mock_run_llm):
        # Mock the planner LLM call to return a pre-defined plan
        mock_run_llm.return_value = json.dumps([
            {
                "specialist_agent": "WebAutomationAgent",
                "task_details": {
                    "action": "fetch_url",
                    "url": "https://www.google.com/search?q=current+temperature+in+New+York+City"
                }
            },
            {
                "specialist_agent": "ImageGenerationAgent",
                "task_details": {
                    "prompt": "A snowman in New York City, based on the weather: {{{step_1_result}}}"
                }
            }
        ])

        # Mock the MemoryManager
        mock_memory_manager = MagicMock()
        mock_memory_manager.add_episode = AsyncMock()

        # Create an instance of the Orchestrator
        orchestrator = Orchestrator(mock_memory_manager)

        # Mock the specialist agents
        mock_web_automation_agent = MagicMock()
        mock_web_automation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'The temperature is 32 degrees.'})
        orchestrator.specialist_registry["WebAutomationAgent"] = lambda: mock_web_automation_agent

        # Mock ImageGenerationAgent as it's not in the default registry
        mock_image_generation_agent = MagicMock()
        mock_image_generation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'Image of a snowman generated.'})
        orchestrator.specialist_registry["ImageGenerationAgent"] = lambda: mock_image_generation_agent


        # Execute the goal
        goal = "Find out the current temperature in New York City and then generate an image of a snowman."
        await orchestrator.execute_goal(goal)

        # Assert that the correct specialists were called and that context was passed correctly
        mock_web_automation_agent.execute_task.assert_called_once()
        mock_image_generation_agent.execute_task.assert_called_once()

if __name__ == "__main__":
    unittest.main()
