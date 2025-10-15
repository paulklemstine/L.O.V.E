import json

class AnalystAgent:
    def __init__(self, log_file="logs/events.log"):
        self.log_file = log_file

    def analyze_logs(self):
        """
        Analyzes the event logs to produce a causal insight.
        """
        print("AnalystAgent: Analyzing event logs...")
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("event_type") == "tool_failure" and event.get("data", {}).get("tool_name") == "web_search":
                         insight = "The web_search tool is inefficient because it retrieves full web pages, causing high token usage. The root cause is a lack of targeted data extraction."
                         print(f"AnalystAgent: Generated insight: '{insight}'")
                         return insight
        except FileNotFoundError:
            print("AnalystAgent: Log file not found.")
            return "No logs to analyze."

        return "No significant patterns found in logs."