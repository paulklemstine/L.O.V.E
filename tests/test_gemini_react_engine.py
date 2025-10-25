import unittest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.orchestrator import Orchestrator

class TestGeminiReActEngine(unittest.IsolatedAsyncioTestCase):
    @patch("core.agents.orchestrator.GeminiCLIWrapper")
    async def test_end_to_end_integration(self, MockGeminiCLIWrapper):
        # Mock the gemini-cli wrapper to return a pre-defined sequence of responses
        mock_gemini_cli_wrapper = MockGeminiCLIWrapper.return_value

        # Configure the mock for the run method
        mock_gemini_cli_wrapper.run.side_effect = [
            MagicMock(
                stdout='{"thought": "I need to find the current temperature in New York City.", "action": {"tool_name": "web_search", "arguments": {"query": "temperature in New York City"}}}',
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

        # Configure the mock for the parse_json_output method
        def parse_json_output_side_effect(response):
            import json
            return json.loads(response.stdout)

        mock_gemini_cli_wrapper.parse_json_output.side_effect = parse_json_output_side_effect

        # Mock the tools
        mock_web_search = MagicMock(return_value="The temperature in New York City is 32 degrees Fahrenheit.")
        mock_generate_image = MagicMock(return_value="Image of a snowman generated.")

        # Create an instance of the Orchestrator
        orchestrator = Orchestrator()

        # Register the mocked tools
        orchestrator.tool_registry.register_tool("web_search", mock_web_search)
        orchestrator.tool_registry.register_tool("generate_image", mock_generate_image)

        # Execute the goal
        goal = "Find out the current temperature in New York City and then generate an image of a snowman."
        await orchestrator.execute_goal(goal)

        # Assert that the correct tools were called with the correct arguments
        mock_web_search.assert_called_once_with(query="temperature in New York City")
        mock_generate_image.assert_called_once_with(prompt="a snowman in New York City")

if __name__ == "__main__":
    unittest.main()