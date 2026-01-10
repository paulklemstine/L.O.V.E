import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage

# Add project root to path
# Resovle root from current file: ./tests/test_agent_graph_system.py -> ../ -> root
project_root = Path(__file__).resolve().parent.parent
# print(f"DEBUG: project_root={project_root}")
sys.path.insert(0, str(project_root))
# print(f"DEBUG: sys.path={sys.path}")

from core.graph import create_deep_agent_graph
import core.nodes.supervisor
import core.agents.planner_agent
import core.nodes.reasoning
import core.nodes.execution

async def verify_complex_goal_routing():
    print("TEST: Verify complex goal routing to planner...")
    app = create_deep_agent_graph()
    
    long_input = "I want you to build a very complex application that does X, Y, and Z and requires a lot of planning and thought."
    inputs = {
        "messages": [HumanMessage(content=long_input)],
        "input": long_input,
        "plan": [],
        "loop_count": 0
    }
    
    with patch.object(core.nodes.supervisor, "run_llm", new_callable=AsyncMock) as mock_llm, \
         patch.object(core.agents.planner_agent, "create_plan", new_callable=AsyncMock) as mock_planner:
        
        mock_llm.return_value = {"result": "<json>{\"next_node\": \"reasoning_node\"}</json>"}
        mock_planner.return_value = {"plan": ["Step 1", "Step 2"]}
        
        visited = []
        async for output in app.astream(inputs):
            for node_name, _ in output.items():
                print(f"  Visited node: {node_name}")
                visited.append(node_name)
                if node_name == "planner_node":
                    break
            if "planner_node" in visited:
                break
        
        if "supervisor" in visited and "planner_node" in visited:
            print("✅ PASS: Routed to planner_node for complex goal.")
        else:
            print(f"❌ FAIL: Did not route to planner_node. Visited: {visited}")

async def verify_tool_usage():
    print("\nTEST: Verify tool usage flow...")
    app = create_deep_agent_graph()
    short_input = "Calculate 2+2"
    inputs = {
        "messages": [HumanMessage(content=short_input)],
        "input": short_input,
        "plan": [],
        "loop_count": 0
    }
    
    with patch.object(core.nodes.supervisor, "run_llm", new_callable=AsyncMock) as mock_sup_llm, \
         patch.object(core.nodes.reasoning, "stream_llm") as mock_stream_llm, \
         patch.object(core.nodes.execution, "_safe_execute_tool", new_callable=AsyncMock) as mock_exec, \
         patch.object(core.nodes.execution, "_get_tool_from_registry") as mock_get_tool:
        
        mock_sup_llm.return_value = {"result": "<json>{\"next_node\": \"reasoning_node\"}</json>"}
        
        async def mock_generator(*args, **kwargs):
            yield 'Thinking... '
            yield '```json {"tool": "calculator", "args": {"expression": "2+2"}} ```'
        mock_stream_llm.return_value = mock_generator()
        
        mock_get_tool.return_value = lambda expression: "4"
        mock_exec.return_value = "4"
        
        visited = []
        async for output in app.astream(inputs):
            for node_name, _ in output.items():
                print(f"  Visited node: {node_name}")
                visited.append(node_name)
                if node_name == "tool_execution_node":
                    break
            if "tool_execution_node" in visited:
                break
        
        if "tool_execution_node" in visited:
            print("✅ PASS: Executed tool.")
        else:
            print(f"❌ FAIL: Did not execute tool. Visited: {visited}")

async def main():
    try:
        await verify_complex_goal_routing()
        await verify_tool_usage()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
