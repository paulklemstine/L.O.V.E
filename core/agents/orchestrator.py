import json
import asyncio
import re
from typing import Dict, List, Any

# Local, dynamic imports for specialist agents are kept for modularity
from core.agents.analyst_agent import AnalystAgent
from core.agents.code_gen_agent import CodeGenerationAgent
from core.agents.rca_agent import RCA_Agent
from core.agents.self_improving_optimizer import SelfImprovingOptimizer
from core.agents.talent_agent import TalentAgent
from core.agents.web_automation_agent import WebAutomationAgent
from core.llm_api import run_llm


def _recursive_substitute(template: Any, context: Dict[str, Any]) -> Any:
    """
    Recursively substitutes placeholders in a nested data structure.
    Placeholders are in the format `{{step_X_result}}`.
    """
    if isinstance(template, str):
        # Corrected regex to properly escape the curly braces for matching
        match = re.fullmatch(r'\{\{(step_\d+_result)\}\}', template)
        if match and match.group(1) in context:
            return context[match.group(1)]
        return template
    elif isinstance(template, dict):
        return {k: _recursive_substitute(v, context) for k, v in template.items()}
    elif isinstance(template, list):
        return [_recursive_substitute(item, context) for item in template]
    else:
        return template

class Orchestrator:
    """
    The Supervisor agent responsible for receiving high-level goals,
    decomposing them into a plan, and orchestrating a team of specialist
    agents to execute the plan.
    """
    def __init__(self):
        """Initializes the Supervisor and its registry of specialist agents."""
        print("Initializing Supervisor Orchestrator...")
        self.specialist_registry = {
            "AnalystAgent": AnalystAgent,
            "CodeGenerationAgent": CodeGenerationAgent,
            "RCA_Agent": RCA_Agent,
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
- **RCA_Agent**: Performs deep Root Cause Analysis on system failures. Expects `task_details` with 'logs', 'memories', and 'graph_summary'.
- **SelfImprovingOptimizer**: Runs a full self-improvement cycle on the codebase. Expects `task_details` with 'task_type' ('improve_module' or 'run_evolution_cycle') and relevant parameters.
- **TalentAgent**: Conducts a full talent scouting, analysis, and engagement cycle. Expects detailed parameters like 'keywords', 'platforms', 'min_score', etc.
- **WebAutomationAgent**: Performs web automation tasks. Expects `task_details` with 'action' ('fetch_url', 'fill_form') and a 'url'.

The high-level goal is: "{goal}"

You must respond with ONLY a JSON array of steps inside a ```json ... ``` block. Each step must be an object with two keys:
1. "specialist_agent": The name of the specialist agent class to use for this step.
2. "task_details": An object containing the parameters for that specialist's `execute_task` method.

You can pass the result of a previous step to a subsequent step using a placeholder string like `{{{{step_X_result}}}}`, where X is the 1-based index of the step.

Example Goal: "Analyze the system logs, and if there's an inefficiency, generate code to fix it."
Example JSON Response:
```json
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
```

Now, generate the plan for the given goal.
"""
        try:
            # run_llm returns a string, not a dictionary
            response_str = await run_llm(prompt, is_source_code=False)

            # More robustly find the JSON block
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_str)
            if not json_match:
                print(f"Supervisor: Failed to find a JSON code block in the LLM response: {response_str}")
                return []

            plan_str = json_match.group(1)
            plan = json.loads(plan_str)
            return plan
        except json.JSONDecodeError as e:
            print(f"Supervisor: Error decoding JSON plan from LLM response: {e}\\nReceived: {plan_str}")
            return []
        except Exception as e:
            print(f"Supervisor: An unexpected error occurred during plan generation: {e}")
            return []

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal, generates a plan, and
        manages the execution of that plan by the specialist agents.
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

            # Substitute context variables from previous steps using the robust recursive function
            task_details = _recursive_substitute(task_details_template, step_results)
            print(f"Task Details: {json.dumps(task_details, indent=2)}")

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

        # --- Persist Learning for RCA Workflows ---
        # This logic is sound and can be kept as is.
        if "critical error" in goal.lower() or "root cause analysis" in goal.lower():
            try:
                from core.memory.memory_manager import MemoryManager
                memory_manager = MemoryManager()

                summary = f"""
                Self-Healing Incident Report:
                - Initial Goal: {goal}
                - Executed Plan: {json.dumps(plan, indent=2)}
                - Final Result: {final_result}
                - Outcome: Success
                """
                # Fire-and-forget call to the async memory addition
                memory_manager.add_memory(summary.strip())
                print("Supervisor: Self-healing incident report logged to agentic memory.")
            except Exception as e:
                print(f"Supervisor: Failed to log self-healing incident to memory. Error: {e}")

        return final_result
