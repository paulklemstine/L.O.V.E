# AGENTS.md - Core Architecture

## Purpose
The `core/` directory contains the fundamental business logic and reasoning engines of L.O.V.E.

## Key Subsystems
### Dependency Injection
- Managed by `dependency_manager.py`.
- **Pattern**: Services (like LLM API, Memory) should be requested via the manager, not instantiated directly in every script.

### Agent Graph
- Defines the flow of control between different cognitive modules.
- See `agent_graph.py` for the DAG definition.

## Invariants
- **Statelessness**: Core logic should minimize local state. Use `love_state.json` or databases for persistence.
- **Async**: Most IO-bound operations must be `async`.

## Anti-patterns
- **Global State**: Do not rely on module-level global variables for state that changes.
- **Circular Imports**: Be very careful with imports in `core/`. Use `TYPE_CHECKING` blocks for type hints if needed.

