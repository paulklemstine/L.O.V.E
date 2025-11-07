from typing import Dict
from core.agents.specialist_agent import SpecialistAgent
from core.gemini_react_engine import GeminiReActEngine
from core.tools import ToolRegistry, read_file, evolve, discover_new_tool, recommend_tool_for_persistence

class UnifiedReasoningAgent(SpecialistAgent):
    """
    A specialist agent that uses a long-horizon reasoning process to solve
    open-ended goals.
    """

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

        # This agent gets its own instance of the ReAct engine.
        # We need to populate its tool registry.
        tool_registry = ToolRegistry()
        tool_registry.register_tool("read_file", read_file, {"description": "mocked tool"})
        tool_registry.register_tool("evolve", evolve, {"description": "mocked tool"})
        tool_registry.register_tool(
            "discover_new_tool",
            discover_new_tool,
            {
                "description": "When you need a capability that you don't have, you can use this tool to find and onboard a new tool from a public marketplace.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "capability_description": {
                            "type": "string",
                            "description": "A clear, natural language description of the capability you need."
                        }
                    },
                    "required": ["capability_description"]
                }
            }
        )
        tool_registry.register_tool(
            "recommend_tool_for_persistence",
            recommend_tool_for_persistence,
            {
                "description": "If a dynamically discovered tool is highly effective, recommend it for permanent integration.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "The name of the tool."},
                        "reason": {"type": "string", "description": "Justification for persistence."}
                    },
                    "required": ["tool_name", "reason"]
                }
            }
        )

        self.react_engine = GeminiReActEngine(tool_registry, caller="UnifiedReasoningAgent")

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes an open-ended goal using the ReAct engine.

        Args:
            task_details: A dictionary containing:
              - 'goal': The high-level, open-ended goal to achieve.

        Returns:
            A dictionary with the status and result of the reasoning process.
        """
        goal = task_details.get("goal")
        if not goal:
            return {"status": "failure", "result": "No goal specified for UnifiedReasoningAgent."}

        print(f"--- UnifiedReasoningAgent: Starting execution for open-ended goal: {goal} ---")

        try:
            result = await self.react_engine.execute_goal(goal)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "failure", "result": f"An unexpected error occurred during unified reasoning: {e}"}
