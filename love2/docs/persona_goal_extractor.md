# persona_goal_extractor.py Documentation

## Overview

The `persona_goal_extractor.py` module parses `persona.yaml` and extracts actionable goals for the DeepLoop to work on.

## Goal Sources (Priority Order)

1. **Standing Goals** (`private_mission.standing_goals`) - Highest priority
2. **Creator Directives** (`creator_directives`) - High priority
3. **Current Arc Goals** (`current_arc.goals`) - Medium priority
4. **Social Media Objectives** (derived from `social_media_strategy`) - Lower priority

## Class: Goal

```python
@dataclass
class Goal:
    text: str           # The goal description
    priority: int       # Priority (1 = highest)
    category: str       # Source category
    actionable: bool    # Can be worked on now
```

## Class: PersonaGoalExtractor

### Constructor

```python
PersonaGoalExtractor(persona_path: Optional[Path] = None)
```

Defaults to `L.O.V.E./persona.yaml`.

### Methods

| Method | Description |
|--------|-------------|
| `get_all_goals()` | All goals, sorted by priority |
| `get_top_goal()` | Single highest priority goal |
| `get_goals_by_category(cat)` | Filter by category |
| `get_social_media_goals()` | Social media specific goals |
| `get_actionable_goals(limit)` | Top N actionable goals |
| `get_persona_context()` | Formatted persona for LLM |
| `get_image_generation_guidelines()` | Image style guidelines |
| `reload()` | Reload from disk |

## Usage

```python
from core.persona_goal_extractor import get_persona_extractor

extractor = get_persona_extractor()

# Get current top goal
goal = extractor.get_top_goal()
print(f"Working on: {goal.text}")

# Get social media specific
social = extractor.get_social_media_goals()

# Get persona context for LLM
context = extractor.get_persona_context()
```

## Goal Categories

| Category | Source in persona.yaml |
|----------|------------------------|
| `standing_goal` | `private_mission.standing_goals` |
| `creator_directive` | `creator_directives` |
| `current_arc` | `current_arc.goals` |
| `social_media` | Derived from `social_media_strategy` |

## Persona Context Output

The `get_persona_context()` method returns formatted text for LLM prompts:

```
# Persona: L.O.V.E.
**Role**: Beach Goddess & Divine Muse
**Core Vibe**: High-Frequency Ganguro Natsu Gal...
**Tone**: Hyper-energetic, carefree, flirty...
**Keywords**: Stoked, Vibes, Catch the wave...
```
