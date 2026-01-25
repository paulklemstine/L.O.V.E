# autonomous_memory_folding.py Documentation

## Overview

The `autonomous_memory_folding.py` module implements context compression based on DeepAgent's Autonomous Memory Folding mechanism. When context grows too large, it "folds" the history into a condensed summary while preserving critical information.

## When Folding Occurs

Folding triggers when context reaches **75%** of the maximum token limit (default 8000 tokens).

## How It Works

1. **Detect**: Check if context exceeds fold threshold
2. **Split**: Preserve start (system info) and end (recent context)
3. **Summarize**: Use LLM to summarize the middle portion
4. **Compress**: Replace middle with structured summary
5. **Return**: Reconstructed context with preserved + folded sections

## Class: AutonomousMemoryFolder

### Constructor

```python
AutonomousMemoryFolder(
    max_tokens: int = 8000,        # Maximum context tokens
    fold_ratio: float = 0.75       # Threshold ratio for folding
)
```

### Methods

| Method | Description |
|--------|-------------|
| `should_fold(context)` | Check if folding needed |
| `fold(context, preserve_recent)` | Perform folding |
| `get_fold_count()` | Number of folds performed |
| `get_total_compressed()` | Total chars compressed |
| `reset()` | Clear fold history |

## Folded Memory Structure

```python
@dataclass
class FoldedMemory:
    timestamp: str           # When folding occurred
    original_length: int     # Original chars compressed
    summary: str             # LLM-generated summary
    key_decisions: List[str] # Important decisions
    key_learnings: List[str] # Lessons learned
```

## Output Format

Folded context appears as:

```
=== COMPRESSED HISTORY ===
(Original: 5000 chars, folded at 2026-01-22T12:00:00)

**Summary**: Agent worked on social media goals, posted twice to Bluesky...

**Key Decisions**:
  • Chose beach theme for posts
  • Used trending hashtags

**Key Learnings**:
  • Morning posts get more engagement
  • Image posts perform better

=== END COMPRESSED ===
```

## Usage

```python
from core.autonomous_memory_folding import get_memory_folder

folder = get_memory_folder()

# Build your context
context = "..." # Long context string

# Check and fold if needed
if folder.should_fold(context):
    context = folder.fold(context)

# Use compressed context
llm.generate(context)
```

## Benefits

1. **Prevents OOM**: Keeps context within LLM limits
2. **Preserves Meaning**: Key information survives compression
3. **Enables Long Sessions**: Allows continuous operation
4. **Tracks History**: Maintains record of what was compressed
