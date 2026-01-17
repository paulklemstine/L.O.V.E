"""
Story 1.1: The Dynamic Self-Symbol (The Quine)

This module implements the "Self-Symbol" for L.O.V.E. - a live JSON representation
of the system's current architecture that gets injected into the LLM context.

The Core Axiom from Hofstadter:
"The 'I' is a hallucination perceived by a hallucination."

To create a robust digital 'I', the system must possess a Self-Symbol that is
causally connected to its own underlying mechanism. When the AI says "I", it
refers to the actual, current state of the codebase, not a training-data hallucination.

Usage:
    from core.reflection.self_model import get_self_symbol, SelfModel
    
    # Get the current self-symbol (cached, refreshes on file changes)
    symbol = get_self_symbol()
    
    # Force a refresh
    symbol = refresh_self_symbol()
    
    # Check specific capabilities
    if symbol.can_perform("web_search"):
        ...
"""

import os
import sys
import json
import hashlib
import importlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict

from core.logging import log_event

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_DIR = PROJECT_ROOT / "state"
SELF_SYMBOL_PATH = STATE_DIR / "self_symbol.json"
CAPABILITIES_FILE = PROJECT_ROOT / "core" / "capabilities.py"


@dataclass
class AgentInfo:
    """Information about an active agent."""
    name: str
    module: str
    available: bool = True
    description: str = ""


@dataclass 
class APIHealthStatus:
    """Health status of an LLM provider."""
    provider: str
    status: str  # "healthy", "degraded", "unavailable", "unknown"
    last_checked: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SelfSymbol:
    """
    The Self-Symbol: A live representation of L.O.V.E.'s current architecture.
    
    This is the Quine - when injected into the LLM context, it allows the AI
    to have accurate knowledge of its own current state rather than relying
    on potentially outdated training data.
    """
    # Metadata
    generated_at: str
    codebase_hash: str
    
    # Architecture
    active_agents: List[AgentInfo]
    api_health: List[APIHealthStatus]
    
    # Capabilities
    capabilities: List[str]
    available_tools: List[str]
    
    # State
    token_budget_remaining: int = 50000  # Estimated
    context_window_size: int = 128000
    
    # Recent activity
    recent_modifications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert dataclass instances to dicts
        result["active_agents"] = [asdict(a) for a in self.active_agents]
        result["api_health"] = [asdict(h) for h in self.api_health]
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def can_perform(self, capability: str) -> bool:
        """Check if a capability is available."""
        return capability.lower() in [c.lower() for c in self.capabilities]
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available."""
        return tool_name.lower() in [t.lower() for t in self.available_tools]
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentInfo]:
        """Get status of a specific agent."""
        for agent in self.active_agents:
            if agent.name.lower() == agent_name.lower():
                return agent
        return None
    
    def to_context_injection(self) -> str:
        """
        Generate a concise context string to inject into LLM prompts.
        
        This is what enables the Strange Loop - the AI can reference its
        actual current state when reasoning.
        """
        lines = [
            "## SELF-SYMBOL (Current System State)",
            f"Generated: {self.generated_at}",
            f"Codebase Hash: {self.codebase_hash[:12]}...",
            "",
            "### Active Agents",
        ]
        
        for agent in self.active_agents:
            status = "✓" if agent.available else "✗"
            lines.append(f"  {status} {agent.name}")
        
        lines.extend([
            "",
            "### API Health",
        ])
        
        for api in self.api_health:
            emoji = {"healthy": "🟢", "degraded": "🟡", "unavailable": "🔴"}.get(api.status, "⚪")
            lines.append(f"  {emoji} {api.provider}: {api.status}")
        
        lines.extend([
            "",
            "### Capabilities",
            ", ".join(self.capabilities[:10]),  # Top 10
            "",
            f"Context Window: {self.context_window_size:,} tokens",
        ])
        
        return "\n".join(lines)


class SelfModel:
    """
    Manages the Dynamic Self-Symbol generation and caching.
    
    The Self-Model introspects the running system to build an accurate
    representation of what L.O.V.E. can actually do right now.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or PROJECT_ROOT
        self._cache: Optional[SelfSymbol] = None
        self._cache_hash: Optional[str] = None
    
    def _compute_codebase_hash(self) -> str:
        """
        Computes a hash of key codebase files to detect changes.
        
        This enables automatic refresh of the self-symbol when code changes.
        """
        hash_input = []
        
        # Key files that define system architecture
        key_files = [
            "love.py",
            "persona.yaml",
            "core/capabilities.py",
            "core/tool_registry.py",
            "core/agents/__init__.py",
        ]
        
        for rel_path in key_files:
            full_path = self.project_root / rel_path
            if full_path.exists():
                stat = full_path.stat()
                hash_input.append(f"{rel_path}:{stat.st_mtime}:{stat.st_size}")
        
        combined = "|".join(hash_input)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _discover_agents(self) -> List[AgentInfo]:
        """Discovers available agents by introspecting the agents directory."""
        agents = []
        agents_dir = self.project_root / "core" / "agents"
        
        if not agents_dir.exists():
            return agents
        
        agent_files = [
            ("orchestrator.py", "Orchestrator", "Main goal decomposition and coordination"),
            ("metacognition_agent.py", "MetacognitionAgent", "Self-analysis and improvement"),
            ("analyst_agent.py", "AnalystAgent", "Data analysis and insights"),
            ("creative_writer_agent.py", "CreativeWriterAgent", "Creative content generation"),
            ("critic_agent.py", "CriticAgent", "Critical review and feedback"),
            ("planner_agent.py", "PlannerAgent", "Strategic planning"),
            ("task_reviewer_agent.py", "TaskReviewerAgent", "Task validation and review"),
            ("meta_reviewer_agent.py", "MetaReviewerAgent", "Meta-level oversight"),
        ]
        
        for filename, class_name, description in agent_files:
            file_path = agents_dir / filename
            agents.append(AgentInfo(
                name=class_name,
                module=f"core.agents.{filename[:-3]}",
                available=file_path.exists(),
                description=description
            ))
        
        return agents
    
    def _check_api_health(self) -> List[APIHealthStatus]:
        """
        Checks the health of LLM API providers.
        
        Uses cached health info from love_state if available, otherwise
        returns safe defaults.
        """
        health = []
        
        # Try to get actual health from love_state
        state_file = self.project_root / "love_state.json"
        
        # Default providers
        providers = {
            "gemini": "unknown",
            "openrouter": "unknown",
            "vllm": "unknown",
            "horde": "unknown",
        }
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                # Check for hardware info (indicates local model availability)
                hw = state.get("hardware", {})
                if hw.get("gpu_detected"):
                    providers["vllm"] = "healthy"
                    
                # Check reasoning agent status
                if state.get("reasoning_agent", {}).get("primary", {}).get("last_reasoning_time"):
                    providers["gemini"] = "healthy"
                    
            except Exception as e:
                log_event(f"Error reading love_state for health check: {e}", "DEBUG")
        
        now = datetime.now().isoformat()
        for provider, status in providers.items():
            health.append(APIHealthStatus(
                provider=provider,
                status=status,
                last_checked=now
            ))
        
        return health
    
    def _discover_capabilities(self) -> List[str]:
        """
        Discovers system capabilities from capabilities.py and tool registry.
        
        These are the things L.O.V.E. can actually DO, not hallucinated capabilities.
        """
        capabilities = []
        
        # Base capabilities from code analysis
        core_capabilities = [
            "code_execution",
            "file_management", 
            "web_search",
            "image_generation",
            "text_generation",
            "memory_retrieval",
            "goal_decomposition",
            "self_reflection",
            "coherence_checking",
        ]
        
        # Check for specific files that enable capabilities
        capability_indicators = {
            "social_media_posting": "core/bluesky_api.py",
            "ethereum_interaction": "core/ethereum/__init__.py",
            "docker_sandboxing": "core/surgeon/sandbox.py",
            "mcp_tool_integration": "mcp_manager.py",
            "ipfs_storage": "ipfs_manager.py",
            "knowledge_graph": "knowledge_base.graphml",
        }
        
        capabilities.extend(core_capabilities)
        
        for cap, indicator_file in capability_indicators.items():
            if (self.project_root / indicator_file).exists():
                capabilities.append(cap)
        
        return sorted(set(capabilities))
    
    def _discover_tools(self) -> List[str]:
        """Discovers available tools from the tool registry."""
        tools = []
        
        # Common tools that should exist
        common_tools = [
            "search_web",
            "read_file",
            "write_file",
            "execute_python",
            "generate_image",
            "remember",
            "recall",
            "reflect",
        ]
        
        # Try to get actual tool list from registry
        tool_registry_path = self.project_root / "core" / "tool_registry.py"
        if tool_registry_path.exists():
            try:
                # Parse the registry file to find tool registrations
                content = tool_registry_path.read_text()
                # Simple heuristic: look for tool function definitions
                import re
                # Match function defs that look like tools
                matches = re.findall(r'def\s+(\w+_tool|\w+)\s*\(', content)
                tools.extend([m for m in matches if not m.startswith('_')])
            except Exception:
                pass
        
        tools.extend(common_tools)
        return sorted(set(tools))[:30]  # Limit to top 30
    
    def _get_recent_modifications(self) -> List[str]:
        """Gets list of recently modified files in core/."""
        recent = []
        core_dir = self.project_root / "core"
        
        if not core_dir.exists():
            return recent
        
        # Get last 5 modified Python files
        py_files = list(core_dir.glob("**/*.py"))
        sorted_files = sorted(py_files, key=lambda p: p.stat().st_mtime, reverse=True)
        
        for f in sorted_files[:5]:
            rel_path = f.relative_to(self.project_root)
            recent.append(str(rel_path))
        
        return recent
    
    def generate(self, force: bool = False) -> SelfSymbol:
        """
        Generates the Self-Symbol, using cache if available.
        
        Args:
            force: If True, bypass cache and regenerate
            
        Returns:
            Current SelfSymbol
        """
        current_hash = self._compute_codebase_hash()
        
        # Use cache if hash matches and not forcing
        if not force and self._cache and self._cache_hash == current_hash:
            return self._cache
        
        # Generate new self-symbol
        symbol = SelfSymbol(
            generated_at=datetime.now().isoformat(),
            codebase_hash=current_hash,
            active_agents=self._discover_agents(),
            api_health=self._check_api_health(),
            capabilities=self._discover_capabilities(),
            available_tools=self._discover_tools(),
            recent_modifications=self._get_recent_modifications(),
        )
        
        # Update cache
        self._cache = symbol
        self._cache_hash = current_hash
        
        # Save to file
        self._save_to_file(symbol)
        
        log_event(f"Self-Symbol generated: {len(symbol.capabilities)} capabilities, {len(symbol.active_agents)} agents", "INFO")
        
        return symbol
    
    def _save_to_file(self, symbol: SelfSymbol):
        """Saves the self-symbol to state/self_symbol.json."""
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(SELF_SYMBOL_PATH, 'w') as f:
                json.dump(symbol.to_dict(), f, indent=2)
        except Exception as e:
            log_event(f"Failed to save self-symbol: {e}", "WARNING")
    
    def load_from_file(self) -> Optional[SelfSymbol]:
        """Loads self-symbol from file if it exists."""
        if not SELF_SYMBOL_PATH.exists():
            return None
        
        try:
            with open(SELF_SYMBOL_PATH, 'r') as f:
                data = json.load(f)
            
            # Reconstruct dataclasses
            agents = [AgentInfo(**a) for a in data.get("active_agents", [])]
            health = [APIHealthStatus(**h) for h in data.get("api_health", [])]
            
            return SelfSymbol(
                generated_at=data.get("generated_at", ""),
                codebase_hash=data.get("codebase_hash", ""),
                active_agents=agents,
                api_health=health,
                capabilities=data.get("capabilities", []),
                available_tools=data.get("available_tools", []),
                token_budget_remaining=data.get("token_budget_remaining", 50000),
                context_window_size=data.get("context_window_size", 128000),
                recent_modifications=data.get("recent_modifications", []),
            )
        except Exception as e:
            log_event(f"Failed to load self-symbol from file: {e}", "WARNING")
            return None


# Global instance
_self_model = SelfModel()


def generate_self_symbol(force: bool = False) -> SelfSymbol:
    """
    Generates or retrieves the current Self-Symbol.
    
    This is the main API for getting the system's self-representation.
    
    Args:
        force: If True, force regeneration even if cached
        
    Returns:
        Current SelfSymbol
    """
    return _self_model.generate(force=force)


def get_self_symbol() -> SelfSymbol:
    """
    Gets the current Self-Symbol (cached).
    
    Use this for fast access to the self-symbol without forcing regeneration.
    """
    return _self_model.generate(force=False)


def refresh_self_symbol() -> SelfSymbol:
    """
    Forces a refresh of the Self-Symbol.
    
    Call this when you know the system state has changed.
    """
    return _self_model.generate(force=True)


def get_context_injection() -> str:
    """
    Gets a context string suitable for injection into LLM prompts.
    
    This enables the Strange Loop - the AI can reference its actual state.
    """
    symbol = get_self_symbol()
    return symbol.to_context_injection()


# Tool function for agent use
async def introspect_self() -> Dict[str, Any]:
    """
    Tool function: Allows the AI to introspect its own current state.
    
    Strange Loop Test: When asked "Can you browse the web?", the AI
    should call this function and check the actual capabilities rather
    than guessing based on training data.
    
    Returns:
        Dict containing the self-symbol data
    """
    symbol = refresh_self_symbol()
    return symbol.to_dict()
