---
description: Run the Intent Gardener to verify code changes against the Intent Layer
---
Run the following command to check your recent changes against the invariants defined in AGENTS.md files. This workflow is useful before pushing changes or when you want to ensure you haven't violated any architectural boundaries.

// turbo
python scripts/intent_gardener.py
