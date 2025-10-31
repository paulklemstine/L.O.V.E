import json
import asyncio
import re
from typing import Dict, List

# Local, dynamic imports for specialist agents
from core.agents.analyst_agent import AnalystAgent
from core.agents.code_gen_agent import CodeGenerationAgent
from core.agents.self_improving_optimizer import SelfImprovingOptimizer
from core.agents.talent_agent import TalentAgent
from core.agents.web_automation_agent import WebAutomationAgent
from core.llm_api import run_llm # Using a direct LLM call for planning

class Orchestrator:
    """
    The Supervisor agent in the hierarchical multi-agent framework.

    This agent is responsible for receiving high-level goals, decomposing them
    into a structured, step-by-step plan, and orchestrating a team of
    specialist agents to execute that plan. It manages the flow of information
    between specialists and ensures the overall goal is achieved.
    """

    def __init__(self):
        """Initializes the Supervisor and its registry of specialist agents."""
        print("Initializing Supervisor Orchestrator...")
        self.specialist_registry = {
            "AnalystAgent": AnalystAgent,
            "CodeGenerationAgent": CodeGenerationAgent,
            "SelfImprovingOptimizer": SelfImprovingOptimizer,
            "TalentAgent": TalentAgent,
            "WebAutomationAgent": WebAutomationAgent,
        }
        print("Supervisor Orchestrator is ready.")

    async def _generate_plan(self, goal: str) -> List[Dict]:
        """
        Uses an LLM to decompose a high-level goal into a structured,
        step-by-step plan for specialist agents.
        """
        print(f"Supervisor: Generating plan for goal: {goal}")

        specialist_list = ", ".join(self.specialist_registry.keys())
        prompt = f"""
You are a Supervisor agent. Your task is to decompose a high-level goal into a step-by-step plan for a team of specialist agents.
The available specialists are: {specialist_list}.

Here are their descriptions:
- **AnalystAgent**: Analyzes logs to find causal insights. Expects `task_details` with a 'logs' key.
- **CodeGenerationAgent**: Generates Python code based on a hypothesis. Expects `task_details` with a 'hypothesis' key.
- **SelfImprovingOptimizer**: Runs a full self-improvement cycle on the codebase. Expects `task_details` with 'task_type' ('improve_module' or 'run_evolution_cycle') and relevant parameters.
- **TalentAgent**: Conducts a full talent scouting, analysis, and engagement cycle. Expects detailed parameters like 'keywords', 'platforms', 'min_score', etc.
- **WebAutomationAgent**: Performs web automation tasks. Expects `task_details` with 'action' ('fetch_url', 'fill_form') and a 'url'.

The high-level goal is: "{goal}"

You must respond with ONLY a JSON array of steps. Each step must be an object with two keys:
1. "specialist_agent": The name of the specialist agent class to use for this step.
2. "task_details": An object containing the parameters for that specialist's `execute_task` method.

You can pass the result of a previous step to a subsequent step using a placeholder string like `{{{{step_X_result}}}}`, where X is the 1-based index of the step.

Example Goal: "Analyze the system logs, and if there's an inefficiency, generate code to fix it."
Example JSON Response:
[
  {{
    "specialist_agent": "AnalystAgent",
    "task_details": {{
      "logs": []
    }}
  }},
  {{
    "specialist_agent": "CodeGenerationAgent",
    "task_details": {{
      "hypothesis": "{{{{step_1_result}}}}"
    }}
  }}
]

Now, generate the plan for the given goal.
"""
        try:
            response_dict = await run_llm(prompt, is_source_code=False)
            response = response_dict.get("result", "")
            # Clean the response to extract only the JSON part
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                print(f"Supervisor: Failed to extract JSON plan from LLM response: {response}")
                return []
            plan_str = json_match.group(0)
            plan = json.loads(plan_str)
            return plan
        except Exception as e:
            print(f"Supervisor: Error during plan generation: {e}")
            return []

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal, generates a plan, and
        manages the execution of that plan by the specialist agents. This
        method embodies the core logic of the Supervisor in the hierarchical
        collaboration framework.
        """
        print(f"\n--- Supervisor received new goal: {goal} ---")

        # 1. Generate Plan
        plan = await self._generate_plan(goal)
        if not isinstance(plan, list) or not plan:
            return "Execution failed: The Supervisor could not generate a valid plan."

        print("Supervisor generated the following plan:")
        print(json.dumps(plan, indent=2))

        # 2. Execute Plan
        step_results = {}
        for i, step in enumerate(plan):
            step_number = i + 1
            specialist_name = step.get("specialist_agent")
            task_details_template = step.get("task_details", {})

            print(f"\n--- Executing Step {step_number}/{len(plan)}: Using {specialist_name} ---")

            # Substitute context variables from previous steps
            try:
                task_details_str = json.dumps(task_details_template)
                for key, value in step_results.items():
                    placeholder = f'{{{{{key}}}}}'
                    # Escape the value to be safely embedded in the JSON string
                    value_escaped = json.dumps(str(value))[1:-1]
                    task_details_str = task_details_str.replace(placeholder, value_escaped)
                task_details = json.loads(task_details_str)
                print(f"Task Details: {json.dumps(task_details, indent=2)}")
            except Exception as e:
                error_msg = f"Plan execution failed at step {step_number}: Failed to substitute context variables. Error: {e}"
                print(error_msg)
                return error_msg

            if specialist_name not in self.specialist_registry:
                error_msg = f"Plan execution failed at step {step_number}: Specialist agent '{specialist_name}' is not registered."
                print(error_msg)
                return error_msg

            try:
                specialist_class = self.specialist_registry[specialist_name]
                specialist_instance = specialist_class()

                result_dict = await specialist_instance.execute_task(task_details)

                print(f"Step {step_number} result: {result_dict}")

                if result_dict.get("status") == "failure":
                    error_msg = f"Plan execution failed at step {step_number} ({specialist_name}): {result_dict.get('result')}"
                    print(error_msg)
                    return error_msg

                step_results[f"step_{step_number}_result"] = result_dict.get("result")

            except Exception as e:
                error_msg = f"An unexpected exception occurred at step {step_number} ({specialist_name}): {e}"
                print(error_msg)
                return error_msg

        final_result = step_results.get(f"step_{len(plan)}_result", "Plan finished with no final result.")
        print(f"\n--- Supervisor finished goal: {goal} ---")
        print(f"Final Result: {final_result}")
        return final_result
