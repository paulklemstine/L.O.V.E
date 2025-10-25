import inspect
from typing import List, Dict, Any, Tuple
from core.gemini_cli_wrapper import GeminiCLIWrapper
from core.tools import ToolRegistry

class GeminiReActEngine:
    """Manages the state of a Thought-Action-Observation loop."""

    def __init__(self, gemini_cli_wrapper: GeminiCLIWrapper, tool_registry: ToolRegistry):
        self.gemini_cli_wrapper = gemini_cli_wrapper
        self.tool_registry = tool_registry
        self.history: List[Tuple[str, str, str]] = []

    async def execute_goal(self, goal: str) -> str:
        """
        Main entry point for the ReAct engine.
        Continues the loop until the Action is a "Finish" action.
        """
        tool_names = self.tool_registry.get_tool_names()
        while True:
            prompt = self._create_prompt(goal, tool_names)
            response = self.gemini_cli_wrapper.run(prompt)
            parsed_response = self.gemini_cli_wrapper.parse_json_output(response)

            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", {})
            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})

            if tool_name == "Finish":
                return f"Goal accomplished. Final thought: {thought}"

            try:
                tool = self.tool_registry.get_tool(tool_name)
                if inspect.iscoroutinefunction(tool):
                    observation = await tool(**arguments)
                else:
                    observation = tool(**arguments)
            except Exception as e:
                observation = f"Error executing tool {tool_name}: {e}"

            self.history.append((thought, action, str(observation)))

    def _create_prompt(self, goal: str, tool_names: List[str]) -> str:
        """Creates the ReAct prompt template."""
        prompt = f"""
You are a large language model tasked with achieving a goal.

Goal: {goal}

You have access to the following tools:
{tool_names}

To achieve the goal, you must output a JSON object with two keys: "thought" and "action".
The "thought" key should contain your reasoning about the current state and what to do next.
The "action" key should contain a JSON object with two keys: "tool_name" and "arguments".
The "tool_name" should be one of the available tools.
The "arguments" should be a dictionary of arguments for the tool.

Here is an example of a successful interaction:

Goal: Find out the current temperature in New York City and then generate an image of a snowman.

History:
(Thought: I need to find the current temperature in New York City. I can use the "web_search" tool for this., Action: {{"tool_name": "web_search", "arguments": {{"query": "temperature in New York City"}}}}, Observation: The temperature in New York City is 32 degrees Fahrenheit.)
(Thought: Now that I have the temperature, I can generate an image of a snowman. I can use the "generate_image" tool for this., Action: {{"tool_name": "generate_image", "arguments": {{"prompt": "a snowman in New York City"}}}}, Observation: Image of a snowman generated.)

Current History:
{self.history}

Based on the goal and the current history, what is the next thought and action?
"""
        return prompt