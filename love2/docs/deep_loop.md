# deep_loop.py Documentation

## Overview

The `deep_loop.py` module implements the main autonomous reasoning loop for L.O.V.E. Version 2. It continuously works towards persona-defined goals using unified agentic reasoning.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  DeepLoop Cycle                  │
├─────────────────────────────────────────────────┤
│ 1. Load persona goals (via PersonaGoalExtractor) │
│ 2. Select highest priority actionable goal       │
│ 3. Reason about goal (LLM call via LLMClient)    │
│ 4. Execute tool actions (via ToolAdapter)        │
│ 5. Update memories (Episodic, Working, Tool)     │
│ 6. Check for memory folding trigger              │
│ 7. Persist state                                 │
│ 8. Sleep (backpressure)                          │
│ 9. GOTO 1                                        │
└─────────────────────────────────────────────────┘
```

## Class: DeepLoop

### Constructor

```python
DeepLoop(
    llm: Optional[LLMClient] = None,
    memory: Optional[MemorySystem] = None,
    persona: Optional[PersonaGoalExtractor] = None,
    folder: Optional[AutonomousMemoryFolder] = None,
    sleep_seconds: float = 30.0,
    max_iterations: Optional[int] = None,
    tools: Optional[Dict[str, Callable]] = None
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `run()` | Start the continuous loop |
| `run_iteration()` | Execute a single iteration |
| `stop()` | Gracefully stop the loop |

### Signal Handling

The loop handles SIGINT and SIGTERM for graceful shutdown, ensuring state is persisted before exit.

## Usage

```python
from core.deep_loop import DeepLoop

# Continuous operation
loop = DeepLoop()
loop.run()

# Test mode (3 iterations)
loop = DeepLoop(max_iterations=3)
loop.run()
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sleep_seconds` | 30.0 | Delay between iterations |
| `max_iterations` | None | Stop after N iterations (None = infinite) |

## LLM Response Format

The LLM must respond with JSON:

```json
{
    "thought": "Reasoning about what to do",
    "action": "tool_name" | "complete" | "skip",
    "action_input": {"param": "value"},
    "reasoning": "Why this action"
}
```
