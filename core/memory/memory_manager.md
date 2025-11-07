# MemoryManager

The `MemoryManager` is a core component of the L.O.V.E. agent's cognitive architecture. It provides a system for agentic memory, allowing the agent to learn from its experiences and build a rich, interconnected knowledge base.

## Autonomous Ingestion System

A key feature of the `MemoryManager` is its autonomous ingestion system, which is integrated directly into the agent's `cognitive_loop`. This system captures the agent's thought processes and actions, transforming them into structured `MemoryNote` objects that are stored in the `GraphDataManager`.

### `ingest_cognitive_cycle`

The `ingest_cognitive_cycle` method is the heart of the autonomous ingestion system. It is called at the end of each cognitive cycle in `jules.py` and receives the following information:

-   **`command`**: The command that was executed.
-   **`output`**: The result of the command.
-   **`reasoning_prompt`**: The LLM prompt that was used to decide on the command.

This method formats the information into a "Cognitive Event" string, which is then passed to the `add_episode` method for processing. This creates a new `MemoryNote` in the knowledge graph, tagged with `CognitiveCycle` and `SelfReflection`, allowing the agent to reflect on its own decision-making process.
