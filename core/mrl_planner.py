import json
from typing import List, Dict, Any
from core.knowledge_graph.graph import KnowledgeGraph
from core.planning import mock_llm_call

class MRLPlanner:
    """
    Analyzes discovered MyRobotLab services in the knowledge base and formulates
    multi-step plans using mrl_call commands to achieve high-level physical-world
    objectives.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def mrl_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Generates a plan of mrl_call commands to achieve a goal.
        """
        services = self.kg.find_services()
        if not services:
            return []

        prompt = (
            f"Given the high-level goal: '{goal}' and the available MyRobotLab services: {services}, "
            "generate a sequence of `mrl_call` commands to achieve the goal. The output should be a JSON array "
            "of objects, where each object has a 'step' number and a 'task' description that is a valid `mrl_call` command."
        )

        try:
            response = mock_llm_call(prompt, purpose="mrl_planning")
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            return []