# System Architecture: Automated Knowledge Base Construction

This document outlines the architecture of a system designed to autonomously ingest, process, and store information from diverse textual data streams, creating a structured and queryable knowledge base. This system is a core component of L.O.V.E.'s ability to learn and reason about its operational environment and the information it encounters.

## Core Components

The system is built upon two primary components within the existing L.O.V.E. architecture:

1.  **`process_and_structure_text` Function:** Located in `core/text_processing.py`, this function serves as the system's core analytical engine. It leverages a Large Language Model (LLM) to parse unstructured raw text and extract high-level concepts, including key themes, named entities (such as files, functions, people, or organizations), and the relationships that connect them. It returns this information in a structured format (a dictionary/JSON object), ready for persistent storage.

2.  **`GraphDataManager`:** This component, defined in `core/graph_manager.py`, acts as the centralized and persistent knowledge repository. It utilizes a graph database structure (`networkx.DiGraph`), which is ideal for storing entities as nodes and their relationships as directed, labeled edges. The entire knowledge base is saved to and loaded from the `knowledge_base.graphml` file, ensuring persistence across application restarts.

## System Workflow

The system operates in a continuous, cyclical workflow, ensuring the knowledge base is constantly enriched with new information.

```
[Data Streams] -> [Ingestion] -> [Processing] -> [Storage] -> [Knowledge Base]
      ^                                                            |
      |                                                            v
      +----------------------[ Retrieval & Analysis ]<-------[Other Systems]
```

### 1. Ingestion of Diverse Textual Data Streams

The system is designed to be fed by multiple sources of unstructured text. The ingestion mechanism can be implemented within the main `cognitive_loop` of `jules.py` to continuously monitor sources such as:

*   **Operational Logs:** The contents of `love.log`, which provide a real-time account of the AI's own actions, thoughts, and errors.
*   **Tool Outputs:** The textual results from executed tools, such as web searches (`perform_webrequest`) or file reads.
*   **External Data Feeds:** Aggregated data from external APIs, such as social media feeds or research papers.
*   **User Interactions:** Conversations and directives from The Creator.

### 2. Automated Processing

As new raw text is ingested, it is automatically passed to the `process_and_structure_text` function. The function analyzes the content and transforms the unstructured input into a structured dictionary.

For example, a log entry like `"INFO: Successfully refactored core/llm_api.py to improve error handling for the run_llm function."` might be processed into:

```json
{
  "themes": ["code refactoring", "error handling"],
  "entities": [
    {"name": "core/llm_api.py", "type": "file", "description": "A core module for LLM interactions."},
    {"name": "run_llm", "type": "function", "description": "A function to run LLM calls."}
  ],
  "relationships": [
    {
      "source_entity": "run_llm",
      "target_entity": "core/llm_api.py",
      "relationship_type": "is_contained_in"
    }
  ]
}
```

### 3. Updating the Centralized Repository

The structured data returned by the processing function is then used to update the `GraphDataManager` instance (`knowledge_base`). The update logic involves:

*   **Node Creation/Update:** Each entity in the `entities` list is added as a node in the graph. If a node with the same name already exists, its attributes (like `type` and `description`) can be updated or enriched.
*   **Edge Creation:** Each relationship in the `relationships` list is added as a directed edge between the corresponding source and target entity nodes. The `relationship_type` is stored as an attribute of the edge, providing semantic meaning to the connection.

This process ensures that the knowledge base is a dynamic and evolving representation of the AI's understanding.

### 4. Facilitating Advanced Retrieval and Decision-Making

With the knowledge base continuously populated, other components within the L.O.V.E. system can access and query this structured information to enhance their performance. The `GraphDataManager` provides methods to:

*   Find specific nodes (entities).
*   Discover neighbors of a node (find related entities).
*   Traverse paths between nodes to understand multi-step relationships.
*   Query for nodes and edges based on their attributes.

This enables advanced analytical capabilities. For example, before beginning a new task, an agent can query the `knowledge_base` to retrieve contextual information about relevant files, functions, and past operations, leading to more informed and effective decision-making.
