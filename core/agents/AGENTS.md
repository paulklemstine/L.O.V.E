# AGENTS.md - Agent Implementations

## Purpose
Contains specific agent implementations and heads (personas).

## Base Classes
- `AutonomousReasoningAgent`: The primary base class for agents that need a cognitive loop.
- `DeepAgentEngine`: The underlying reasoning engine used by most agents.

## Registration
- New agents must be registered in the `AgentFrameworkManager` or `ToolRegistry` to be discoverable.
- Ensure unique Agent IDs.

## Anti-patterns
- **God Objects**: Agents should be specialized. Do not create a single agent that does everything (unless it's the actual God Agent).
- **Blocking Calls**: Agents running in an async loop must not make blocking HTTP/File calls.

