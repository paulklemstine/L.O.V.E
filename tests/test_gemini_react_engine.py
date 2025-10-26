import unittest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.orchestrator import Orchestrator

class TestGeminiReActEngine(unittest.IsolatedAsyncioTestCase):
    @patch("core.llm_api.GeminiCLIWrapper")
    async def test_end_to_end_integration(self, MockGeminiCLIWrapper):
        # Mock the gemini-cli wrapper to return a pre-defined sequence of responses
        mock_gemini_cli_wrapper = MockGeminiCLIWrapper.return_value

        # Configure the mock for the run method
        mock_gemini_cli_wrapper.run.side_effect = [
            MagicMock(
                stdout='{"thought": "I need to find the current temperature in New York City.", "action": {"tool_name": "perform_webrequest", "arguments": {"url": "https://www.google.com/search?q=temperature+in+New+York+City"}}}',
                stderr="",
                return_code=0,
            ),
            MagicMock(
                stdout='{"thought": "Now that I have the temperature, I can generate an image of a snowman.", "action": {"tool_name": "generate_image", "arguments": {"prompt": "a snowman in New York City"}}}',
                stderr="",
                return_code=0,
            ),
            MagicMock(
                stdout='{"thought": "I have completed all the steps.", "action": {"tool_name": "Finish", "arguments": {}}}',
                stderr="",
                return_code=0,
            ),
        ]


        # Mock the tools
        mock_perform_webrequest = MagicMock(return_value="The temperature in New York City is 32 degrees Fahrenheit.")
        mock_generate_image = MagicMock(return_value="Image of a snowman generated.")

        # Create an instance of the Orchestrator
        orchestrator = Orchestrator()

        # Register the mocked tools
        orchestrator.tool_registry.register_tool("perform_webrequest", mock_perform_webrequest, {"description": "mocked tool"})
        orchestrator.tool_registry.register_tool("generate_image", mock_generate_image, {"description": "mocked tool"})

        # Execute the goal
        goal = "Find out the current temperature in New York City and then generate an image of a snowman."
        await orchestrator.execute_goal(goal)

        # Assert that the correct tools were called with the correct arguments
        mock_perform_webrequest.assert_called_once_with(url="https://www.google.com/search?q=temperature+in+New+York+City")
        mock_generate_image.assert_called_once_with(prompt="a snowman in New York City")

if __name__ == "__main__":
    unittest.main()