# AGENTS.md - Agent Implementations

## Purpose
Contains specific agent implementations and heads (personas).

## Base Classes
- `AutonomousReasoningAgent`: The primary base class for agents that need a cognitive loop.
- `DeepAgentEngine`: The underlying reasoning engine used by most agents.

## Registration
- New agents must be registered in the `AgentFrameworkManager` or `ToolRegistry` to be discoverable.
- Ensure unique Agent IDs.
