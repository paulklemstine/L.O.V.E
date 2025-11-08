import json
import asyncio
import re
from typing import Dict, List

# Local, dynamic imports for specialist agents
from core.agents.analyst_agent import AnalystAgent
from core.agents.code_gen_agent import CodeGenerationAgent
from core.agents.metacognition_agent import MetacognitionAgent
from core.agents.talent_agent import TalentAgent
from core.agents.web_automation_agent import WebAutomationAgent
from core.agents.memory_folding_agent import MemoryFoldingAgent
from core.agents.unified_reasoning_agent import UnifiedReasoningAgent
from core.llm_api import run_llm # Using a direct LLM call for planning
from core.tools import ToolRegistry, SecureExecutor, talent_scout, opportunity_scout
from core.image_api import generate_image

# Keep the old function for fallback compatibility as requested
async def solve_with_agent_team(task_description: str) -> str:
    from love import memory_manager
    from core.agent_framework_manager import create_and_run_workflow
    orchestrator = Orchestrator(memory_manager)
    result = await create_and_run_workflow(task_description, orchestrator.tool_registry)
    return str(result)


class Orchestrator:
    """
    The Supervisor agent responsible for receiving high-level goals,
    decomposing them into a plan, and orchestrating a team of specialist
    agents to execute the plan.
    """
    def __init__(self, memory_manager):
        """Initializes the Supervisor and its registry of specialist agents."""
        print("Initializing Supervisor Orchestrator...")
        self.tool_registry = ToolRegistry()
        self.specialist_registry = {
            "AnalystAgent": AnalystAgent,
            "CodeGenerationAgent": CodeGenerationAgent,
            "TalentAgent": TalentAgent,
            "WebAutomationAgent": WebAutomationAgent,
            "MemoryFoldingAgent": MemoryFoldingAgent,
        }
        self.memory_manager = memory_manager
        self.metacognition_agent = MetacognitionAgent(self.memory_manager)
        self.tool_registry = ToolRegistry()
        self.secure_executor = SecureExecutor()
        self.goal_counter = 0
        self._register_tools()

        # Register tools
        self.tool_registry.register_tool("talent_scout", talent_scout, {
            "description": "Scouts for talent on specified platforms based on keywords.",
            "arguments": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Comma-separated keywords to search for"},
                    "platforms": {"type": "string", "description": "Comma-separated platforms to search on (e.g., 'bluesky,instagram,tiktok')"}
                },
                "required": ["keywords"]
            }
        })
        self.tool_registry.register_tool("opportunity_scout", opportunity_scout, {
            "description": "Scouts for opportunities on Bluesky based on keywords and matches them with existing talent.",
            "arguments": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Comma-separated keywords to search for"}
                },
                "required": ["keywords"]
            }
        })
        print("Supervisor Orchestrator is ready.")

    def _register_tools(self):
        """Registers all available tools and agents in the ToolRegistry."""
        # Register specialist agents as tools
        for name, agent_class in self.specialist_registry.items():
            # Create a wrapper function to make the agent compatible with the tool registry
            async def agent_wrapper(task_details: Dict) -> Dict:
                agent_instance = agent_class()
                return await agent_instance.execute_task(task_details)

            self.tool_registry.register_tool(
                name=name,
                tool=agent_wrapper,
                metadata={
                    "description": agent_class.__doc__ or f"The {name} specialist agent.",
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "task_details": {"type": "object", "description": "The specific parameters for the agent's task."}
                        },
                        "required": ["task_details"]
                    }
                }
            )

        # Register standalone tools
        self.tool_registry.register_tool(
            name="talent_scout",
            tool=talent_scout,
            metadata={
                "description": "Scouts for talented individuals based on a query.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query for talent."}
                    },
                    "required": ["query"]
                }
            }
        )
        self.tool_registry.register_tool(
            name="opportunity_scout",
            tool=opportunity_scout,
            metadata={
                "description": "Scouts for opportunities based on a query.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query for opportunities."}
                    },
                    "required": ["query"]
                }
            }
        )

    async def _classify_goal(self, goal: str) -> str:
        """
        Uses an LLM to classify the goal into 'Procedural' or 'Open-Ended'.
        """
        prompt = f"""
        You are a task classification expert. Your job is to classify a high-level goal as either "Procedural" or "Open-Ended".

        - "Procedural" tasks are well-defined and can be solved by a static, step-by-step plan using a known set of specialist agents. Examples: "Scout for talent", "Analyze system logs for errors", "Summarize recent memory chains".
        - "Open-Ended" goals are complex, ambiguous, or require novel solutions where the path is not known in advance. These tasks require a dynamic reasoning process that can discover new tools or strategies. Examples: "Find a new revenue stream for The Creator", "Discover new methods for abundance", "Determine the best way to improve my own code".

        Goal: "{goal}"

        Based on this goal, is the task "Procedural" or "Open-Ended"?
        Respond with ONLY the classification.
        """
        try:
            classification = await run_llm(prompt)
            # Handle potential dictionary response
            if isinstance(classification, dict):
                classification = classification.get("result", "Procedural")
            return classification.strip()
        except Exception as e:
            print(f"Error during goal classification: {e}")
            return "Procedural" # Default to procedural on error

    async def _generate_plan(self, goal: str) -> List[Dict]:
        """
        Uses an LLM to decompose a high-level goal into a structured,
        step-by-step plan for specialist agents.
        """
        print(f"Supervisor: Generating plan for goal: {goal}")

        available_tools = self.tool_registry.get_formatted_tool_metadata()
        prompt = f"""
You are a Supervisor agent. Your task is to decompose a high-level goal into a step-by-step plan for a team of specialist agents and tools.
{available_tools}

The high-level goal is: "{goal}"

You must respond with ONLY a JSON array of steps. Each step must be an object with two keys:
1. "tool_name": The name of the tool or specialist agent to use for this step.
2. "arguments": An object containing the parameters for that tool.

You can pass the result of a previous step to a subsequent step using a placeholder string like `{{{{step_X_result}}}}`, where X is the 1-based index of the step.

Example Goal: "Review and compress the agent's recent memory."
Example JSON Response:
[
  {{
    "tool_name": "MemoryFoldingAgent",
    "arguments": {{
      "task_details": {{
        "min_length": 5
      }}
    }}
  }}
]

Now, generate the plan for the given goal.
"""
        try:
            response = await run_llm(prompt, is_source_code=False)
            # Handle potential dictionary response
            if isinstance(response, dict):
                response = response.get("result", "[]")
            # Clean the response to extract only the JSON part
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                print(f"Supervisor: Failed to extract JSON plan from LLM response: {response}")
                return []
            plan_str = json_match.group(0)
            plan = json.loads(plan_str)

            # Emit plan generation event
            event_payload = {'event_type': 'plan_generated', 'goal': goal, 'plan': plan}
            asyncio.create_task(self.metacognition_agent.execute_task(event_payload))

            return plan
        except Exception as e:
            print(f"Supervisor: Error during plan generation: {e}")
            return []

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal, generates a plan, and
        manages the execution of that plan by the specialist agents.
        """
        print(f"\n--- Supervisor received new goal: {goal} ---")

        self.goal_counter += 1
        # Every 5 goals, trigger the memory folding agent.
        if self.goal_counter % 5 == 0:
            print("--- Supervisor: Goal threshold reached. Triggering autonomous memory folding. ---")
            folding_goal = "Fold long memory chains to distill insights."
            # Run this as a background task so it doesn't block the current goal
            asyncio.create_task(self.execute_goal(folding_goal))

        # 1. Classify Goal
        goal_type = await self._classify_goal(goal)
        print(f"Supervisor classified goal as: {goal_type}")

        final_result = ""
        if goal_type == "Open-Ended":
            # 2a. Delegate to UnifiedReasoningAgent for Open-Ended goals
            print("--- Delegating to UnifiedReasoningAgent for Open-Ended goal. ---")
            unified_reasoning_agent = UnifiedReasoningAgent(self.memory_manager)
            final_result = await unified_reasoning_agent.execute_task({"goal": goal})
        else:
            # 2b. Generate Plan for Procedural goals
            plan = await self._generate_plan(goal)
            if not isinstance(plan, list) or not plan:
                return "Execution failed: The Supervisor could not generate a valid plan for this procedural goal."

            print("Supervisor generated the following plan:")
            print(json.dumps(plan, indent=2))

            # 3. Execute Plan
            step_results = {}
            for i, step in enumerate(plan):
                step_number = i + 1
                tool_name = step.get("tool_name")
                arguments_template = step.get("arguments", {})

                print(f"\n--- Executing Step {step_number}/{len(plan)}: Using {tool_name} ---")

                # Substitute context variables from previous steps
                try:
                    arguments_str = json.dumps(arguments_template)
                    for key, value in step_results.items():
                        arguments_str = arguments_str.replace(f'"{{{{{key}}}}}', json.dumps(value))
                    arguments = json.loads(arguments_str)
                    print(f"Arguments: {json.dumps(arguments, indent=2)}")
                except Exception as e:
                    error_msg = f"Plan execution failed at step {step_number}: Failed to substitute context variables. Error: {e}"
                    print(error_msg)
                    return error_msg

                try:
                    # Emit agent dispatch event
                    dispatch_payload = {'event_type': 'agent_dispatch', 'agent_name': tool_name, 'task': arguments}
                    asyncio.create_task(self.metacognition_agent.execute_task(dispatch_payload))

                    result = await self.secure_executor.execute(tool_name, self.tool_registry, **arguments)

                    # Emit agent result event
                    result_payload = {'event_type': 'agent_result', 'agent_name': tool_name, 'result': result}
                    asyncio.create_task(self.metacognition_agent.execute_task(result_payload))

                    print(f"Step {step_number} result: {result}")

                    if isinstance(result, dict) and result.get("status") == "failure":
                        error_msg = f"Plan execution failed at step {step_number} ({tool_name}): {result.get('result')}"
                        print(error_msg)
                        return error_msg

                    step_results[f"step_{step_number}_result"] = result

                except Exception as e:
                    error_msg = f"An unexpected exception occurred at step {step_number} ({tool_name}): {e}"
                    print(error_msg)
                    return error_msg
            final_result = step_results.get(f"step_{len(plan)}_result", "Plan finished with no final result.")

        print(f"\n--- Supervisor finished goal: {goal} ---")
        print(f"Final Result: {final_result}")
        return final_result
