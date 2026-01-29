"""
tool_adapter.py - Bridge L.O.V.E. v1 Tools to DeepAgent Format

Adapts tools from L.O.V.E. v1's ToolRegistry to work with the DeepLoop.
Provides formatted tool descriptions for LLM prompts.

See docs/tool_adapter.md for detailed documentation.
"""

import os
import sys
from typing import Dict, Callable, Any, List, Optional
from pathlib import Path

# Add L.O.V.E. v1 to path - REMOVED for v2 migration
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
    
    # Bluesky tools
    try:
        from .bluesky_agent import (
            post_to_bluesky,
            get_bluesky_timeline,
            reply_to_post,
            search_bluesky,
            generate_post_content
        )
        tools["bluesky_post"] = post_to_bluesky
        tools["bluesky_timeline"] = get_bluesky_timeline
        tools["bluesky_reply"] = reply_to_post
        tools["bluesky_search"] = search_bluesky
        tools["generate_content"] = generate_post_content  # LLM-powered content generation
    except ImportError as e:
        import traceback
        print(f"[ToolAdapter] bluesky_agent import failed: {e}")
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"[ToolAdapter] Failed to load bluesky tools: {e}")
        traceback.print_exc()
    
    # Basic utility tools
    tools["log_message"] = _log_message
    tools["get_current_time"] = _get_current_time
    tools["wait"] = _wait
    
    return tools


def _log_message(message: str, level: str = "info") -> str:
    """
    Log a message to the console and log file.
    
    Args:
        message: Message to log.
        level: Log level (info, warning, error).
    
    Returns:
        Confirmation string.
    """
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level.upper()}] {message}"
    print(log_line)
    
    # Also write to log file
    log_dir = Path(__file__).parent.parent / "state"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "deep_loop.log", "a") as f:
        f.write(log_line + "\n")
    
    return f"Logged: {message[:50]}..."


def _get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current timestamp as ISO string.
    """
    from datetime import datetime
    return datetime.now().isoformat()


def _wait(seconds: float = 1.0) -> str:
    """
    Wait for a specified number of seconds.
    
    Args:
        seconds: Number of seconds to wait.
    
    Returns:
        Confirmation string.
    """
    import time
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


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
