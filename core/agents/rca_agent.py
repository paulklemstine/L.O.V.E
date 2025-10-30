import json
from typing import Dict, List
from core.agents.specialist_agent import SpecialistAgent
from core.llm_api import run_llm

class RCA_Agent(SpecialistAgent):
    """
    A specialist agent that performs Root Cause Analysis on system failures.
    """
    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Analyzes logs and memory to find the root cause of a failure.

        Args:
            task_details: A dictionary containing:
                - 'logs': A list of recent log entries.
                - 'memories': A list of recent agent memories.
                - 'graph_summary': A summary of the knowledge graph state.

        Returns:
            A dictionary containing the structured RCA report.
        """
        logs = task_details.get('logs', [])
        memories = task_details.get('memories', [])
        graph_summary = task_details.get('graph_summary', 'Not available.')

        log_str = "\\n".join(logs)
        memories_str = "\\n".join(memories)

        prompt = f"""
You are an expert Root Cause Analysis agent. Your mission is to analyze system data to understand the fundamental cause of a critical failure.

Here is the data you have collected:

--- RECENT LOGS ---
{log_str}

--- RECENT MEMORIES (Agent's thought-action-observation loop) ---
{memories_str}

--- KNOWLEDGE GRAPH SUMMARY ---
{graph_summary}

Based on all of this information, please perform a root cause analysis and generate a structured report. The report must be a JSON object with the following schema:
{{
    "hypothesized_root_cause": "A detailed, specific hypothesis about the single, most likely root cause of the failure. Be precise.",
    "confidence_score": "A float between 0.0 and 1.0 representing your confidence in the hypothesis.",
    "recommended_actions": [
        "A list of concrete, actionable steps to fix the underlying issue.",
        "For code changes, provide a clear description of the required modification."
    ]
}}

Your response MUST be only the raw JSON object, with no other text, comments, or formatting.
"""

        try:
            response_str = await run_llm(prompt, is_source_code=False)
            report = json.loads(response_str)
            return {
                "status": "success",
                "result": report
            }
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode LLM response into JSON. Error: {e}\\nResponse: {response_str}"
            print(error_msg)
            return {"status": "failure", "result": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during RCA analysis: {e}"
            print(error_msg)
            return {"status": "failure", "result": error_msg}
