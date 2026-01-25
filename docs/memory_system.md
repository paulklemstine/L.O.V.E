# memory_system.py Documentation

## Overview

The `memory_system.py` module implements a brain-inspired memory architecture based on DeepAgent patterns. It provides three types of memory that work together to enable intelligent reasoning.

## Memory Types

### 1. Episodic Memory

**Purpose**: High-level log of key events, decisions, and sub-task completions.

**Use Cases**:
- Tracking completed goals
- Recording significant actions
- Logging errors and milestones

**Key Methods**:
- `add_event(event_type, summary, details)` - Add a new event
- `get_recent(count)` - Get recent events
- `get_by_type(event_type)` - Filter by event type

### 2. Working Memory

**Purpose**: Current sub-goal and near-term plans (the "scratchpad").

**Contents**:
- Current goal being worked on
- Sub-goals for decomposed tasks
- Step-by-step plan
- Last action and result
- Iteration counter

**Key Methods**:
- `set_goal(goal, sub_goals)` - Set current goal
- `set_plan(steps)` - Set execution plan
- `record_action(action, result)` - Record last action
- `complete_sub_goal(sub_goal)` - Mark sub-goal done

### 3. Tool Memory

**Purpose**: Consolidated tool interactions, enabling learning from experience.

**Tracks**:
- Success/failure counts per tool
- Execution times
- Last errors
- Learned patterns

**Key Methods**:
- `record_usage(tool_name, success, time_ms, error)` - Log tool use
- `get_reliable_tools(min_success_rate)` - Find high-performing tools
- `add_pattern(pattern)` - Record learned pattern

## MemorySystem Class

Unified interface combining all memory types with persistence.

### Persistence

State is saved to `love2/state/` as JSON files:
- `episodic_memory.json`
- `working_memory.json`
- `tool_memory.json`

### Key Methods

| Method | Description |
|--------|-------------|
| `save()` | Persist all memories to disk |
| `get_full_context()` | Get combined context for LLM |
| `record_action(...)` | Record action across all memories |
| `record_goal_start(goal)` | Mark goal as started |
| `record_goal_complete(goal)` | Mark goal as complete |

## Usage

```python
from core.memory_system import MemorySystem

memory = MemorySystem()

# Record goal start
memory.record_goal_start("Post to Bluesky")

# Record action
memory.record_action(
    tool_name="bluesky_post",
    action='{"text": "Hello!"}',
    result="Posted successfully",
    success=True,
    time_ms=150.0
)

# Get context for LLM
context = memory.get_full_context()
```
