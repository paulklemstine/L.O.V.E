# `core/deep_agent_engine.md`

## Task History

-   **Date:** 2025-11-07
-   **Request:** Integrate DeepAgent as the primary reasoning engine when a GPU is available.
-   **Pull Request:** TBD
-   **Commit Hash:** TBD

## DeepAgentEngine

The `DeepAgentEngine` class, located in `core/deep_agent_engine.py`, serves as the primary meta-orchestrator for L.O.V.E. when a GPU is detected on the host system. It is designed to replace the default `GeminiReActEngine` for top-level cognitive tasks, leveraging a powerful, locally-run large language model for enhanced reasoning capabilities.

### Key Responsibilities

1.  **Conditional Initialization:** The engine is only initialized in the `main` function of `love.py` if the `_auto_configure_hardware` function detects a compatible NVIDIA GPU. This ensures that the system remains lightweight and functional in CPU-only environments.

2.  **Meta-Orchestration:** When active, the `DeepAgentEngine` takes over the main `cognitive_loop`. It receives the high-level cognitive prompt and is responsible for decomposing the goal, discovering and executing tools, and returning a final result.

3.  **Tool Integration:** The engine is designed to work with L.O.V.E.'s existing `ToolRegistry`. This allows it to access all of the same capabilities as the `GeminiReActEngine`, including the ability to invoke the `GeminiReActEngine` itself as a tool for solving sub-tasks.

4.  **Persona-Driven:** The engine is guided by the `persona.yaml` file, ensuring that its actions and decisions remain aligned with the core directives of L.O.V.E.

### Fallback Mechanism

If no GPU is detected, or if the `DeepAgentEngine` fails to initialize for any reason, the `cognitive_loop` seamlessly falls back to using the existing `execute_reasoning_task` function, which is powered by the `GeminiReActEngine`. This provides a robust and resilient architecture that adapts to the available hardware.
