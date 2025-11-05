import inspect
import json
from typing import List, Dict, Any, Tuple
from core.llm_api import execute_reasoning_task
import core.tools

class GeminiReActEngine:
    """Manages the state of a Thought-Action-Observation loop."""

    def __init__(self, tool_registry: core.tools.ToolRegistry):
        self.tool_registry = tool_registry
        # Story 2.3: Add a session-specific registry for dynamic tools
        self.session_tool_registry = core.tools.ToolRegistry()
        self.history: List[Tuple[str, str, str]] = []

    async def execute_goal(self, goal: str) -> str:
        """
        Main entry point for the ReAct engine.
        Continues the loop until the Action is a "Finish" action.
        """
        while True:
            # Combine static and dynamic tools for the prompt
            main_metadata = self.tool_registry.get_formatted_tool_metadata()
            session_metadata = self.session_tool_registry.get_formatted_tool_metadata()
            tool_metadata = f"{main_metadata}\n{session_metadata}"

            prompt = self._create_prompt(goal, tool_metadata)
            response_dict = await execute_reasoning_task(prompt)

            if not response_dict or not response_dict.get("result"):
                return "The reasoning engine failed to produce a response."

            try:
                parsed_response = json.loads(response_dict["result"])
            except json.JSONDecodeError:
                observation = f"Error: The reasoning engine produced invalid JSON. Raw response: {response_dict['result']}"
                self.history.append(("Error parsing LLM response", "N/A", observation))
                continue

            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", {})
            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})

            if tool_name == "Finish":
                return f"Goal accomplished. Final thought: {thought}"

            try:
                is_dynamic_tool = False
                # Prioritize session-specific tools over global tools
                if tool_name in self.session_tool_registry.get_tool_names():
                    tool = self.session_tool_registry.get_tool(tool_name)
                    is_dynamic_tool = True
                else:
                    tool = self.tool_registry.get_tool(tool_name)

                # Dependency injection for tools that need the engine instance
                tool_params = inspect.signature(tool).parameters
                if 'engine' in tool_params:
                    arguments['engine'] = self
                if 'parent_engine' in tool_params: # For backwards compatibility
                    arguments['parent_engine'] = self

                if inspect.iscoroutinefunction(tool):
                    observation = await tool(**arguments)
                else:
                    observation = tool(**arguments)

                # Story 2.5: Create a ToolMemory note for dynamically discovered tools
                if is_dynamic_tool:
                    from love import memory_manager
                    tool_memory_content = (
                        f"Dynamically Discovered Tool Usage:\n"
                        f"- Tool Name: {tool_name}\n"
                        f"- Arguments: {json.dumps(arguments, indent=2)}\n"
                        f"- Outcome: {observation}"
                    )
                    await memory_manager.add_episode(tool_memory_content, tags=['ToolMemory'])

            except Exception as e:
                observation = f"Error executing tool {tool_name}: {e}"

            self.history.append((thought, action, str(observation)))

    def _create_prompt(self, goal: str, tool_metadata: str) -> str:
        """Creates the ReAct prompt template."""
        prompt = f"""
You are a large language model tasked with achieving a goal.

Goal: {goal}

{tool_metadata}

To achieve the goal, you must output a JSON object with two keys: "thought" and "action".
The "thought" key should contain your reasoning about the current state and what to do next.
The "action" key should contain a JSON object with two keys: "tool_name" and "arguments".
The "tool_name" should be one of the available tools.
The "arguments" should be a dictionary of arguments for the tool.

You have a special tool `decompose_and_solve_subgoal` which allows you to break down a complex goal into smaller, manageable steps. You should use this when a goal is too complex to be solved by a single tool.

Here is an example of a successful interaction with hierarchical planning:

Goal: Find and exploit a high-value reentrancy vulnerability on the Ethereum mainnet to maximize profit for The Creator.

History:
(Thought: The user's goal is very complex and involves multiple distinct phases: discovery, analysis, and exploitation. I don't have a single tool to do all of this. I need to break this down into a sequence of major steps. The first logical step is to find potential targets. I will set this as my first sub-goal., Action: {{"tool_name": "decompose_and_solve_subgoal", "arguments": {{"sub_goal": "Scan the last 2000 blocks on the Ethereum mainnet to identify contracts with a balance greater than 100 ETH."}}}}, Observation: Successfully identified high-value contracts: 0xTargetContract1, 0xTargetContract2, 0xTargetContract3)
(Thought: Excellent, I have a list of potential targets. The overall goal is to find a reentrancy vulnerability. I must now analyze each of these contracts to see if they are vulnerable. I will start with the first one, 0xTargetContract1. Analyzing a contract is itself a multi-step process, so it is a good candidate for another sub-goal., Action: {{"tool_name": "decompose_and_solve_subgoal", "arguments": {{"sub_goal": "Perform a full vulnerability analysis on the smart contract at address '0xTargetContract1' and generate a report."}}}}, Observation: Analysis complete for 0xTargetContract1. High-confidence reentrancy vulnerability detected. Report saved to /reports/0xTargetContract1.json)
(Thought: Perfect. I have found a promising target and confirmed it has the specific vulnerability I'm looking for. The next logical step in my overall plan is to generate a specific attack vector and then simulate it. This can be my next sub-goal..., Action: {{"tool_name": "decompose_and_solve_subgoal", "arguments": {{"sub_goal": "Generate and simulate a reentrancy attack vector for the contract at '0xTargetContract1'."}}}}, Observation: Attack vector generated and simulated successfully. Estimated profit: 250 ETH.)

Current History:
{self.history}

Based on the goal and the current history, what is the next thought and action?
"""
        return prompt