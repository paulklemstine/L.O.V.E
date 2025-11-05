import json
from typing import Dict
from core.agents.specialist_agent import SpecialistAgent

class AnalystAgent(SpecialistAgent):
    """
    A specialist agent that analyzes logs and memory to find causal insights.
    """
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Analyzes data based on the task type.
        """
        task_type = task_details.get("task_type", "analyze_logs")

        if task_type == "analyze_logs":
            return self._analyze_logs(task_details)
        elif task_type == "analyze_tool_memory":
            return self._analyze_tool_memory()
        else:
            return {"status": "failure", "result": f"Unknown task type: {task_type}"}

    def _analyze_logs(self, task_details: Dict) -> Dict:
        """
        Analyzes logs to find causal insights.
        """
        logs = task_details.get("logs")
        if not logs:
            return {"status": "failure", "result": "No logs provided."}

        # ... (existing log analysis logic)
        return {"status": "success", "result": "No significant patterns found in logs."}

    def _analyze_tool_memory(self) -> Dict:
        """
        Scans ToolMemory nodes to find recommendations for persistent tools.
        """
        if not self.memory_manager:
            return {"status": "failure", "result": "MemoryManager not available."}

        print("AnalystAgent: Analyzing ToolMemory for persistence recommendations...")

        tool_memory_nodes = self.memory_manager.graph_data_manager.query_nodes("tags", "ToolMemory")

        for node_id in tool_memory_nodes:
            node_data = self.memory_manager.graph_data_manager.get_node(node_id)
            if "Tool Persistence Recommendation" in node_data.get("content", ""):
                insight = (
                    f"Insight: A dynamically discovered tool has been recommended for persistence. "
                    f"The recommendation is: {node_data.get('content')}"
                )
                print(f"AnalystAgent: Generated insight: '{insight}'")
                return {"status": "success", "result": insight}

        return {"status": "success", "result": "No tool persistence recommendations found."}
