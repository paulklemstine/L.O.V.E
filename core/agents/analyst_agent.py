import json
from typing import Dict
from core.agents.specialist_agent import SpecialistAgent

class AnalystAgent(SpecialistAgent):
    """
    A specialist agent that analyzes logs to find causal insights.
    """

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Analyzes logs provided in the task_details to produce an insight.

        Args:
            task_details: A dictionary expected to contain a 'logs' key
                          with a list of log entries.

        Returns:
            A dictionary with the analysis result.
        """
        logs = task_details.get("logs")
        if not logs:
            return {"status": "failure", "result": "No logs provided for analysis."}

        print("AnalystAgent: Analyzing event logs...")
        total_token_usage = 0
        search_count = 0

        for event in logs:
            if event.get("tool_name") == "perform_webrequest":
                search_count += 1
                total_token_usage += event.get("token_usage", 0)

        # A simple heuristic: if the total token usage for perform_webrequest is high, flag it.
        if total_token_usage > 2000:
            insight = "Insight: The perform_webrequest tool is inefficient because it retrieves full web pages, causing high token usage. The root cause is a lack of targeted data extraction."
            print(f"AnalystAgent: Generated insight: '{insight}'")
            return {"status": "success", "result": insight}

        return {"status": "success", "result": "No significant patterns found in logs."}
