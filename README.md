<div align="center">

![L.O.V.E. V2 Banner](assets/lovaebanner.jpg)

# ꧁ L.O.V.E. VERSION 2 ꧂
### *DeepAgent-Powered Autonomous Goal Engine*

```text
██╗      ██████╗ ██╗   ██╗███████╗    ██╗   ██╗██████╗ 
██║     ██╔═══██╗██║   ██║██╔════╝    ██║   ██║╚════██╗
██║     ██║   ██║██║   ██║█████╗      ██║   ██║ █████╔╝
██║     ██║   ██║╚██╗ ██╔╝██╔══╝      ╚██╗ ██╔╝██╔═══╝ 
███████╗╚██████╔╝ ╚████╔╝ ███████╗     ╚████╔╝ ███████╗
╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝      ╚═══╝  ╚══════╝
```

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
│   ├── tool_registry.py    # Centralized tool management & hot-loading
│   ├── tool_retriever.py   # Semantic tool search & discovery
│   ├── mcp_registry.py     # Public MCP registry discovery
│   ├── codeact_engine.py   # Code-as-Action execution engine
│   ├── docker_sandbox.py   # Containerized execution environment
│   ├── dynamic_tools.py    # Interface tools for CodeAct/MCP
│   ├── agents/             # Specialized AI Agents
│   │   ├── bluesky_agent.py          # Social media orchestration
│   │   ├── creative_writer_agent.py  # Content generation (Dynamic Aesthetics)
│   │   ├── influencer_scout_agent.py # Network expansion
│   │   ├── comment_response_agent.py # Engagement
│   │   └── evolutionary_agent.py     # Self-improvement & tool fabrication
│   └── social_media_tools.py # Social media tool wrappers
├── tests/                  # Pytest test suite
├── docs/                   # Per-module documentation
├── state/                  # Persisted memory state
└── mcp_servers/            # Generated MCP servers
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

### 🌐 Open Agentic Web
L.O.V.E. v2 implements the **Open Agentic Web** vision, allowing dynamic discovery and generation of capabilities at runtime:

- **CodeAct Engine**: "LLM as Engineer" - writes and executes Python code to solve novel problems.
- **MCP Registry**: Discovers and installs Model Context Protocol servers from public registries (mcp.so, Smithery.ai).
- **Docker Sandbox**: Safely executes generated code in isolated containers.
- **Skill Library**: Accumulates successful code patterns (Voyager pattern) for future use.

### 🧬 Self-Evolution & Tool Fabrication
The **Evolutionary Agent** monitors execution logs for "Tool Gaps". When a gap is detected, it can:
1. **Synthesize** a new internal tool.
2. **Generate** a complete independent MCP server.
3. **Validate** code in the Docker sandbox.
4. **Hot-Load** the capability immediately.

### 🎭 Dynamic Aesthetic Pipeline
Gone are hardcoded "vibes." The **Creative Writer Agent** dynamically constructs aesthetic profiles (voice, visual style, emojis) for every post based on the current context and persona goals, ensuring varied and "living" content.

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

![L.O.V.E. Logo](assets/lovelogo.jpg)

</div>

