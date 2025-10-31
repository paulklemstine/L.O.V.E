import unittest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.agents.orchestrator import Orchestrator

class TestGeminiReActEngine(unittest.IsolatedAsyncioTestCase):
    @patch("core.agents.orchestrator.run_llm", new_callable=AsyncMock)
    async def test_end_to_end_integration(self, mock_run_llm):
        # Mock the planner LLM call to return a pre-defined plan in the correct format
        mock_run_llm.return_value = {
            "result": json.dumps([
                {
                    "specialist_agent": "WebAutomationAgent",
                    "task_details": {
                        "action": "fetch_url",
                        "url": "https://www.google.com/search?q=weather+new+york+city"
                    }
                },
                {
                    "specialist_agent": "ImageGenerationAgent",
                    "task_details": {
                        "prompt": "A snowman in New York City, based on the weather: {{step_1_result}}"
                    }
                }
            ])
        }

        # Create an instance of the Orchestrator
        orchestrator = Orchestrator()

        # Mock the specialist agents
        mock_web_automation_agent = MagicMock()
        mock_web_automation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'The temperature is 32 degrees.'})

        mock_image_generation_agent = MagicMock()
        mock_image_generation_agent.execute_task = AsyncMock(return_value={'status': 'success', 'result': 'Image of a snowman generated.'})

        # Inject mock specialist classes (as lambdas returning instances) into the registry
        orchestrator.specialist_registry = {
            "WebAutomationAgent": lambda: mock_web_automation_agent,
            "ImageGenerationAgent": lambda: mock_image_generation_agent,
        }

        # Execute the goal
        goal = "Find out the current temperature in New York City and then generate an image of a snowman."
        await orchestrator.execute_goal(goal)

        # Assert that the correct specialists were called with the correct, context-substituted task details
        mock_web_automation_agent.execute_task.assert_called_once_with({
            "action": "fetch_url",
            "url": "https://www.google.com/search?q=weather+new+york+city"
        })

        mock_image_generation_agent.execute_task.assert_called_once_with({
            "prompt": "A snowman in New York City, based on the weather: The temperature is 32 degrees."
        })

if __name__ == "__main__":
    unittest.main()
