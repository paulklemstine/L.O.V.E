# AGENTS.md - Tools & Capabilities

## Purpose
This directory contains the tools that agents can use to interact with the world (and the system). Tools are wrapped functions exposed to the LLM.

## Invariants
- **Type Hinting**: All tool arguments must be type-hinted. The `ToolRegistry` relies on inspections to generate schemas.
- **Docstrings**: Every tool MUST have a Google-style docstring explaining its purpose, arguments, and return values. This is what the LLM sees.

## Anti-patterns
- **Side Effects**: Tools should be relatively contained. Avoid tools that secretly modify global system state without a clear name indicating so.
- **Over-complexity**: If a tool takes more than 5 complex arguments, break it down.
