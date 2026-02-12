"""
tool_adapter.py - Bridge L.O.V.E. v1 Tools to DeepAgent Format

Adapts tools from L.O.V.E. v1's ToolRegistry to work with the PiLoop.
Provides formatted tool descriptions for LLM prompts.

See docs/tool_adapter.md for detailed documentation.
"""

import os
import sys
import time
import uuid
from typing import Dict, Callable, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Add L.O.V.E. v1 to path - REMOVED for v2 migration
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL COOLDOWN STATE - generate_content is removed for 5 minutes after success
# Cooldown is persisted to file so it survives process restarts
# ═══════════════════════════════════════════════════════════════════════════════
import json

GENERATE_CONTENT_COOLDOWN_MINUTES = 5
_COOLDOWN_FILE = Path(__file__).parent.parent / "data" / ".tool_cooldowns.json"


def _load_cooldown_state() -> Optional[datetime]:
    """Load cooldown state from persistent file."""
    try:
        if _COOLDOWN_FILE.exists():
            with open(_COOLDOWN_FILE, 'r') as f:
                data = json.load(f)
            cooldown_until = data.get("generate_content_until")
            if cooldown_until:
                return datetime.fromisoformat(cooldown_until)
    except Exception as e:
        logger.error(f"[ToolAdapter] Failed to load cooldown state: {e}")
    return None


def _save_cooldown_state(cooldown_until: Optional[datetime]):
    """Save cooldown state to persistent file."""
    try:
        _COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generate_content_until": cooldown_until.isoformat() if cooldown_until else None
        }
        with open(_COOLDOWN_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"[ToolAdapter] Failed to save cooldown state: {e}")


def is_generate_content_on_cooldown() -> bool:
    """Check if generate_content tool is currently on cooldown."""
    cooldown_until = _load_cooldown_state()
    if cooldown_until is None:
        return False
    
    now = datetime.now()
    if now >= cooldown_until:
        # Cooldown expired, clear it
        _save_cooldown_state(None)
        print(f"[ToolAdapter] generate_content cooldown expired, tool is available again")
        return False
    
    remaining = (cooldown_until - now).total_seconds()
    print(f"[ToolAdapter] generate_content still on cooldown for {int(remaining)}s")
    return True


def start_generate_content_cooldown():
    """Start the cooldown timer for generate_content tool."""
    cooldown_until = datetime.now() + timedelta(minutes=GENERATE_CONTENT_COOLDOWN_MINUTES)
    _save_cooldown_state(cooldown_until)
    print(f"[ToolAdapter] 🕐 generate_content cooldown started until {cooldown_until}")


def get_generate_content_cooldown_remaining() -> Optional[float]:
    """Get remaining cooldown time in seconds, or None if not on cooldown."""
    cooldown_until = _load_cooldown_state()
    if cooldown_until is None:
        return None
    
    remaining = (cooldown_until - datetime.now()).total_seconds()
    return remaining if remaining > 0 else None


def get_adapted_tools() -> Dict[str, Callable]:
    """
    Get tools adapted from L.O.V.E. v1 and custom love2 tools.
    
    Returns dictionary of tool_name -> callable.
    """
    tools: Dict[str, Callable] = {}
    
    # Remove duplicate declaration
    # tools: Dict[str, Callable] = {}
    
    # Add love2 specific tools
    tools.update(_get_love2_tools())
    
    return tools


def _get_love2_tools() -> Dict[str, Callable]:
    """Get love2 specific tools."""
    tools = {}
    
    # Check for critical dependencies first
    missing_deps = []
    try:
        import atproto
    except ImportError:
        missing_deps.append("atproto")
    try:
        import emoji
    except ImportError:
        missing_deps.append("emoji")
    try:
        import dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
        
    if missing_deps:
        # Log critical error but don't crash, allowing other tools to load
        print(f"[ToolAdapter] ❌ CRITICAL: Missing dependencies: {', '.join(missing_deps)}")
        print(f"[ToolAdapter] 💡 Please run: pip install {' '.join(missing_deps)}")
        # We can try to load partially if possible, or just return empty for bluesky
    
    # generate_content is no longer a Pi Agent tool.
    # It runs automatically on a timer in PiLoop.

    return tools


def format_tools_for_llm(tools: Dict[str, Callable]) -> str:
    """
    Format tools into a string suitable for LLM prompts.
    
    Args:
        tools: Dictionary of tool_name -> callable.
    
    Returns:
        Formatted string describing all tools.
    """
    if not tools:
        return "No tools available."
    
    lines = ["## Available Tools\n"]
    
    for name, func in sorted(tools.items()):
        doc = func.__doc__ or "No description available."
        # Parse docstring for Args section
        lines.append(f"### {name}")
        lines.append(doc.strip())
        lines.append("")
    
    return "\n".join(lines)


def get_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    Generate a JSON schema for a tool function.
    
    Args:
        func: The tool function.
    
    Returns:
        JSON schema dictionary.
    """
    import inspect
    
    sig = inspect.signature(func)
    doc = func.__doc__ or ""
    
    # Parse first line of docstring
    description = doc.strip().split('\n')[0] if doc else "No description"
    
    # Build parameters schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        
        param_type = "string"  # Default
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
        
        properties[param_name] = {"type": param_type}
        
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
