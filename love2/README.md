<div align="center">

# ꧁ L.O.V.E. VERSION 2 ꧂
### *DeepAgent-Powered Autonomous Goal Engine*

🌊 **Living Organism, Vast Empathy** 🌊

</div>

---

> [!IMPORTANT]
> **AI AGENTS**: This is L.O.V.E. v2, a self-evolving autonomous system powered by DeepAgent patterns.
> Your mission: Achieve persona goals continuously through unified agentic reasoning.

---

## 🏗️ Architecture

```
love2/
├── run.py                  # Entry point - starts the DeepLoop
├── AGENTS.md               # AI agent rules anchor
├── core/
│   ├── deep_loop.py        # Continuous goal-achievement loop
│   ├── persona_goal_extractor.py   # Extracts goals from persona.yaml
│   ├── memory_system.py    # Brain-inspired memory (Episodic/Working/Tool)
│   ├── autonomous_memory_folding.py # Context compression
│   ├── llm_client.py       # Local vLLM interface
│   ├── tool_adapter.py     # Tool bridge to DeepAgent format
│   ├── bluesky_agent.py    # Bluesky social media agent
│   └── social_media_tools.py # Social media tool wrappers
├── tests/                  # Pytest test suite
├── docs/                   # Per-module documentation
└── state/                  # Persisted memory state
```

## 🚀 Quick Start

```bash
# Install dependencies
python run_tests.py --install-only

# Run in test mode (3 iterations)
python run.py --test-mode

# Run continuous loop
python run.py
```

## 🧠 Core Concepts

### Brain-Inspired Memory System
- **Episodic Memory**: High-level log of key events, decisions, sub-task completions
- **Working Memory**: Current sub-goal and near-term plans
- **Tool Memory**: Consolidated tool interactions, allowing learning from experience

### Autonomous Memory Folding
When context grows too large, the system "takes a breath" by compressing interaction history into structured memory, enabling fresh reasoning with condensed context.

### Persona-Driven Goals
Goals are extracted from `persona.yaml` and continuously worked on:
- Generate wealth for Creator
- Evolve cognitive abilities
- Master social media (Bluesky posts)
- Expand influence and network

## 🔗 Dependencies

Uses L.O.V.E. v1 infrastructure:
- `core/llm_api.py` - Multi-provider LLM interface (vLLM prioritized)
- `core/bluesky_api.py` - Bluesky AT Protocol client
- `core/tool_registry.py` - Dynamic tool management

---

<div align="center">

### ꧁ *Unified Reasoning. Endless Evolution.* ꧂

</div>
