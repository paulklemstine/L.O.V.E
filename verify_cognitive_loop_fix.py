
import asyncio
import sys
import os
from unittest.mock import MagicMock

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Mock environment variables
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = "dummy_token"

from core.strategic_reasoning_engine import StrategicReasoningEngine

# Mock love_state and knowledge_base
love_state = {
    "autopilot_goal": "Test Goal",
    "autopilot_history": []
}
knowledge_base = MagicMock()
knowledge_base.get_node.return_value = {}
knowledge_base.get_neighbors.return_value = []
knowledge_base.query_nodes.return_value = []
knowledge_base.get_all_edges.return_value = []
knowledge_base.get_all_nodes.return_value = []

async def mock_cognitive_loop_snippet():
    print("Simulating cognitive loop 'strategize' command handling...")
    try:
        strategic_engine = StrategicReasoningEngine(knowledge_base, love_state)
        # This is the line we fixed
        plan = await strategic_engine.generate_strategic_plan()
        output = "Generated Strategic Plan:\n" + "\n".join(f"- {step}" for step in plan)
        print("Success! Output generated:")
        print(output)
    except TypeError as e:
        print(f"FAIL: TypeError caught: {e}")
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")

if __name__ == "__main__":
    asyncio.run(mock_cognitive_loop_snippet())
