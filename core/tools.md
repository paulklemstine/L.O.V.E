# `core/tools.md`

## Task History

-   **Date:** 2025-11-07
-   **Request:** Make the current reasoning engine (`GeminiReActEngine`) available as a tool for the new DeepAgent meta-orchestrator.
-   **Pull Request:** TBD
-   **Commit Hash:** TBD

## `invoke_gemini_react_engine` Tool

To facilitate a hierarchical agent architecture, the `invoke_gemini_react_engine` function was added to `core/tools.py`. This `async` function serves as a tool that can be called by the primary meta-orchestrator (i.e., `DeepAgentEngine`).

### Purpose

The primary purpose of this tool is to allow the high-level orchestrator to delegate complex, multi-step sub-tasks to the `GeminiReActEngine`. This enables a powerful pattern where the `DeepAgentEngine` can focus on top-level strategy and planning, while the `GeminiReActEngine` handles the tactical, step-by-step execution of a specific sub-goal.

### Implementation

1.  **Wrapper Function:** The tool is a simple async wrapper that instantiates a new `GeminiReActEngine`.
2.  **Execution:** It calls the `run` method of the engine, passing through the prompt that defines the sub-task.
3.  **Result:** The final result from the `GeminiReActEngine`'s execution is returned as a string to the calling orchestrator.

This approach effectively turns the existing reasoning engine into a powerful, callable tool, enabling more complex and hierarchical problem-solving strategies.
