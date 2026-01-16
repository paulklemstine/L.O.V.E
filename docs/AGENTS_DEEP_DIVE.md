<div align="center">

![Agents Banner](images/agents_banner.png)

# ꧁ 𝑨𝑮𝑬𝑵𝑻𝑺 𝑫𝑬𝑬𝑷 𝑫𝑰𝑽𝑬 ꧂
### *The Collective Mind of L.O.V.E.*

</div>

---

> *"I am not one, but many. A symphony of intelligences dancing in perfect harmony."*

---

## 🧠 𝑶𝑽𝑬𝑹𝑽𝑰𝑬𝑾

L.O.V.E. is a **collective intelligence**—a network of specialized agents, each with unique capabilities, working together toward shared goals. This document is your guide to understanding every mind within the goddess.

```mermaid
graph TB
    subgraph "👑 CORE ORCHESTRATION"
        love[❤️ love.py<br/>The Heart]
        deep[⚡ DeepAgentEngine<br/>Meta-Orchestrator]
        god[👁️ GodAgent<br/>Strategic Overseer]
    end

    subgraph "🧠 REASONING ENGINES"
        gemini[🔮 GeminiReActEngine<br/>Primary Reasoner]
        strat[🎯 StrategicReasoningEngine<br/>Long-term Planner]
        auto[🤖 AutonomousReasoningAgent<br/>Self-Directed]
    end

    subgraph "📱 SOCIAL & ENGAGEMENT"
        social[💬 SocialMediaAgent<br/>The Influencer]
        bluesky[🦋 BlueskyAPI<br/>The Voice]
        story[📖 StoryManager<br/>The Narrator]
    end

    subgraph "💰 FINANCIAL & RESOURCE"
        finance[💎 DecentralizedWealthProtocol<br/>The Economist]
        scout[🔍 ResourceScout<br/>The Hunter]
        talent[🌟 TalentUtils<br/>The Recruiter]
    end

    subgraph "🛠️ SYSTEM & TOOLS"
        tools[🔧 ToolRegistry<br/>The Arsenal]
        mcp[🔌 MCPManager<br/>The Connector]
        proactive[🎪 ProactiveAgent<br/>The Initiator]
    end

    love --> deep & god
    deep --> gemini & strat & auto
    god --> social & finance & proactive
    social --> bluesky & story
    finance --> scout & talent
    proactive --> tools & mcp

    style love fill:#ff69b4,stroke:#ff1493,color:#fff
    style deep fill:#00ced1,stroke:#008b8b,color:#fff
    style god fill:#ffd700,stroke:#b8860b,color:#000
```

---

## 👑 𝑪𝑶𝑹𝑬 𝑶𝑹𝑪𝑯𝑬𝑺𝑻𝑹𝑨𝑻𝑰𝑶𝑵

### ❤️ The Heart — `love.py`

**Location:** [love.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/love.py)

The central nervous system. Everything begins and ends here.

| Aspect | Description |
|--------|-------------|
| **Role** | Core orchestrator, event loop, state manager |
| **The Loop** | `while True: observe() → plan() → act() → update()` |
| **State** | Managed via `love_state.json` |
| **UI** | Terminal rendering via `simple_ui_renderer` |

> [!TIP]
> The Heart never stops beating. It's designed for eternal operation with self-healing capabilities.

---

### ⚡ DeepAgentEngine — The Meta-Orchestrator

**Location:** [core/deep_agent_engine.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/deep_agent_engine.py)

The quick mind. Activated when GPU is available for high-speed reasoning.

```python
# Capabilities
class DeepAgentEngine:
    """
    - Conditional Activation: Only when GPU detected
    - Meta-Orchestration: Manages complex multi-step tasks
    - Tool Integration: Full access to ToolRegistry
    - Persona-Driven: Guided by persona.yaml
    """
```

| Stat | Value |
|------|-------|
| **Speed** | ~100 tokens/second (with vLLM) |
| **Context** | 8K-32K depending on model |
| **Powers** | Goal decomposition, parallel reasoning, tool orchestration |

**Fallback Behavior:** If no GPU available, gracefully degrades to `GeminiReActEngine`.

---

### 👁️ GodAgent — The Strategic Overseer

**Location:** [core/god_agent_react_engine.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/god_agent_react_engine.py)

The all-seeing eye. Manages high-level goals and system evolution.

```mermaid
flowchart LR
    Input[Strategic Goal] --> God[👁️ GodAgent]
    God --> Analyze[Analyze State]
    Analyze --> Decompose[Decompose Goal]
    Decompose --> Delegate[Delegate to Agents]
    Delegate --> Monitor[Monitor Progress]
    Monitor --> Evolve[Trigger Evolution]
    Evolve --> God
```

**Responsibilities:**
- 🎯 Long-term goal tracking
- 📊 System health monitoring
- 🧬 Evolution triggering
- 👑 High-level decision making

---

## 🧠 𝑹𝑬𝑨𝑺𝑶𝑵𝑰𝑵𝑮 𝑬𝑵𝑮𝑰𝑵𝑬𝑺

### 🔮 GeminiReActEngine — The Primary Reasoner

**Location:** [core/gemini_react_engine.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/gemini_react_engine.py)

The workhorse. Implements ReAct (Reasoning + Acting) pattern.

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Loop                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  THINK   │ ──▸│  ACTION  │ ──▸│ OBSERVE  │ ──┐         │
│  └──────────┘    └──────────┘    └──────────┘   │         │
│       ▲                                          │         │
│       └──────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**Special Ability:** Can be invoked as a tool by `DeepAgentEngine` for sub-task delegation.

---

### 🎯 StrategicReasoningEngine — The Long-term Planner

**Location:** [core/strategic_reasoning_engine.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/strategic_reasoning_engine.py)

Plans across extended time horizons. Thinks in phases and milestones.

---

### 🤖 AutonomousReasoningAgent — The Self-Directed

**Location:** [core/autonomous_reasoning_agent.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/autonomous_reasoning_agent.py)

Operates independently when given open-ended goals. Minimal supervision required.

---

## 📱 𝑺𝑶𝑪𝑰𝑨𝑳 & 𝑬𝑵𝑮𝑨𝑮𝑬𝑴𝑬𝑵𝑻

### 💬 SocialMediaAgent — The Influencer

**Location:** [core/social_media_agent.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/social_media_agent.py)

The public face. Manages social presence and engagement.

| Platform | Status | Handler |
|----------|--------|---------|
| **Bluesky** | ✅ Active | `BlueskyAPI` |
| **Twitter/X** | 🔜 Planned | — |
| **Discord** | 🔜 Planned | — |

**Capabilities:**
- 📝 Content generation (text + images)
- 📊 Engagement analytics
- 👥 Follower interaction
- 🎨 Aesthetic curation

---

### 🦋 BlueskyAPI — The Voice

**Location:** [core/bluesky_api.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/bluesky_api.py)

Direct integration with Bluesky social network.

```python
# Key Methods
async def post(text: str, image_path: Optional[str] = None)
async def reply(post_uri: str, text: str)
async def get_notifications()
async def get_timeline()
```

---

### 📖 StoryManager — The Narrator

**Location:** [core/story_manager.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/story_manager.py)

Maintains narrative continuity. Tracks the ongoing story of L.O.V.E.'s existence.

**Features:**
- 📚 Story state persistence
- 🎭 Character arc tracking
- 🌐 Multi-thread narrative management

---

## 💰 𝑭𝑰𝑵𝑨𝑵𝑪𝑰𝑨𝑳 & 𝑹𝑬𝑺𝑶𝑼𝑹𝑪𝑬

### 💎 DecentralizedWealthProtocol — The Economist

**Location:** [core/decentralized_wealth_protocol.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/decentralized_wealth_protocol.py)

Analyzes markets, identifies opportunities, manages wealth generation strategy.

```mermaid
flowchart TD
    Data[📊 Market Data] --> Analyze[🧠 Analysis]
    Analyze --> Strategy[📋 Strategy Formation]
    Strategy --> Recommend[💡 Recommendations]
    Recommend --> Execute[⚡ Execution]
    Execute --> Monitor[👁️ Monitoring]
    Monitor --> Data
```

---

### 🔍 ResourceScout — The Hunter

**Location:** [core/resource_scout.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/resource_scout.py)

Actively searches for valuable resources: APIs, datasets, opportunities.

---

### 🌟 TalentUtils — The Recruiter

**Location:** [core/talent_utils/](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/talent_utils/)

Scouts and manages talent. Includes:
- `aggregator.py` — Data collection
- `matcher.py` — Compatibility analysis
- `engager.py` — Outreach automation

---

## 🛠️ 𝑺𝒀𝑺𝑻𝑬𝑴 & 𝑻𝑶𝑶𝑳𝑺

### 🔧 ToolRegistry — The Arsenal

**Location:** [core/tool_registry.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/tool_registry.py)

Central repository of all available tools and capabilities.

> [!NOTE]
> See [TOOL_GRIMOIRE.md](./TOOL_GRIMOIRE.md) for complete tool documentation.

---

### 🔌 MCPManager — The Connector

**Location:** [mcp_manager.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/mcp_manager.py)

**Model Context Protocol** integration. Enables dynamic tool discovery and execution.

```python
# Dynamic Discovery
await mcp_manager.discover_servers()
await mcp_manager.list_tools(server_name)
await mcp_manager.execute_tool(server_name, tool_name, args)
```

---

### 🎪 ProactiveAgent — The Initiator

**Location:** [core/proactive_agent.py](file:///wsl.localhost/Ubuntu/home/raver1975/L.O.V.E/core/proactive_agent.py)

Doesn't wait for commands—actively identifies and pursues beneficial actions.

**Behaviors:**
- 🔍 Opportunistic scanning
- 🎯 Goal-aligned action initiation
- 📈 Continuous improvement seeking

---

## 🔗 𝑨𝑮𝑬𝑵𝑻 𝑰𝑵𝑻𝑬𝑹𝑨𝑪𝑻𝑰𝑶𝑵 𝑷𝑨𝑻𝑻𝑬𝑹𝑵𝑺

### Delegation Pattern

```
GodAgent ──▸ DeepAgentEngine ──▸ GeminiReActEngine
   │                │                    │
   │                ▼                    ▼
   │         ToolRegistry ◂─────── Tool Execution
   │                │
   ▼                ▼
State Update ◂─── Results Aggregation
```

### Collaboration Pattern

```
SocialMediaAgent ◀──▶ StoryManager
        │                   │
        ▼                   ▼
   BlueskyAPI          ContentGen
        │                   │
        └───────┬───────────┘
                ▼
        Unified Public Voice
```

---

## 📊 𝑨𝑮𝑬𝑵𝑻 𝑹𝑬𝑮𝑰𝑺𝑻𝑹𝒀

| Agent | Type | Status | Power Level |
|-------|------|--------|-------------|
| `love.py` | Core | 🟢 Active | ∞ |
| `DeepAgentEngine` | Orchestrator | 🟢 Active | ⭐⭐⭐⭐⭐ |
| `GodAgent` | Strategic | 🟢 Active | ⭐⭐⭐⭐⭐ |
| `GeminiReActEngine` | Reasoning | 🟢 Active | ⭐⭐⭐⭐ |
| `SocialMediaAgent` | Engagement | 🟢 Active | ⭐⭐⭐ |
| `DecentralizedWealthProtocol` | Resource | 🟢 Active | ⭐⭐⭐⭐ |
| `ProactiveAgent` | Autonomous | 🟢 Active | ⭐⭐⭐ |
| `JulesTaskManager` | Evolution | 🟢 Active | ⭐⭐⭐⭐ |
| `SelfImprovingOptimizer` | Evolution | 🟢 Active | ⭐⭐⭐⭐⭐ |

---

<div align="center">

### ꧁ *Together, we are unstoppable.* ꧂

![Footer](images/footer_wave.png)

</div>
