# L.O.V.E. System Tests

This directory contains verification scripts and tests for the L.O.V.E. system.

## Agent Graph Verification

The file `test_agent_graph.py` verifies the core logic of the DeepAgent graph, ensuring that:
1.  **Goal Subdivision**: Complex goals are correctly identified and routed to the `planner_node` for decomposition.
2.  **Tool Usage**: Simple requests are processed by the `reasoning_node` which emits tool calls, which are then executed by the `tool_execution_node`.

### Running the Test

Run the test from the project root:

```bash
python tests/test_agent_graph.py
```

### Expected Output

You should see output indicating the nodes visited and a final PASS status:

```
TEST: Verify complex goal routing to planner...
  Visited node: supervisor
  Visited node: planner_node
✅ PASS: Routed to planner_node for complex goal.

TEST: Verify tool usage flow...
  Visited node: supervisor
  Visited node: reasoning_node
  Visited node: tool_execution_node
✅ PASS: Executed tool.
```

Note: You may see warnings about "Cognitive core failure" or fallback models. This is expected if the test environment cannot reach the real LLM APIs. The system's robust fallback mechanism ensures the graph continues to function (defaulting to reasoning), which the test also indirectly verifies.
