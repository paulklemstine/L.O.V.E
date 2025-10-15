import json

class AnalystAgent:
    @staticmethod
    def analyze_logs(logs: list) -> str:
        """
        Analyzes a list of event logs to produce a causal insight.
        """
        print("AnalystAgent: Analyzing event logs...")
        total_token_usage = 0
        search_count = 0

        for event in logs:
            if event.get("tool_name") == "web_search":
                search_count += 1
                total_token_usage += event.get("token_usage", 0)

        # A simple heuristic: if the total token usage for web_search is high, flag it.
        if total_token_usage > 2000:
            insight = "Insight: The web_search tool is inefficient because it retrieves full web pages, causing high token usage. The root cause is a lack of targeted data extraction."
            print(f"AnalystAgent: Generated insight: '{insight}'")
            return insight

        return "No significant patterns found in logs."