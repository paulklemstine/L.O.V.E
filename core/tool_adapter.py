"""
tool_adapter.py - Bridge L.O.V.E. v1 Tools to DeepAgent Format

Adapts tools from L.O.V.E. v1's ToolRegistry to work with the DeepLoop.
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
    
    # Bluesky tools with cooldown - tool is REMOVED from registry during cooldown
    try:
        from .bluesky_agent import (
            generate_post_content
        )
        
        # Check if tool is in cooldown (should not be added to tools)
        if not is_generate_content_on_cooldown():
            def generate_content_wrapper(topic: str = None, **kwargs) -> Dict[str, Any]:
                """
                Generate and post content to Bluesky.
                
                After a successful post, this tool will be removed for 5 minutes
                to allow time for engagement before posting again.
                
                Args:
                    topic: Optional topic/theme for the post.
                
                Returns:
                    Dict with post details including post_uri on success.
                """
                result = generate_post_content(topic=topic, **kwargs)
                
                # If successful, start cooldown
                if result.get("success") and result.get("posted", True):
                    start_generate_content_cooldown()
                
                return result
            
            # Copy docstring for tool discovery
            generate_content_wrapper.__doc__ = generate_post_content.__doc__
            tools["generate_content"] = generate_content_wrapper
        else:
            print(f"[ToolAdapter] generate_content is on cooldown, not adding to tools")
    
    except ImportError as e:
        import traceback
        print(f"[ToolAdapter] bluesky_agent import failed: {e}")
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"[ToolAdapter] Failed to load bluesky tools: {e}")
        traceback.print_exc()

    # Pi Agent tool
    async def ask_pi_agent(prompt: str, timeout: float = 600.0) -> str:
        """
        Send a prompt to the Pi Agent and get a response.
        
        The Pi Agent is a powerful coding assistant that can help with
        complex tasks, code generation, and answering questions.
        
        Args:
            prompt: The message/question to send to the Pi Agent.
            timeout: Maximum time to wait for response in seconds (default: 600s).
        
        Returns:
            The Pi Agent's response text.
        """
        import asyncio
        from .pi_rpc_bridge import get_pi_bridge
        
        logger.info(f"[ask_pi_agent] Prompt: {prompt}")
        logger.info("[ask_pi_agent] 🧠 Pi Agent is thinking...")
        
        bridge = get_pi_bridge()
        response_text = []
        response_complete = asyncio.Event()
        callback_id = f"ask_pi_{uuid.uuid4().hex[:8]}"
        
        # For streaming to logs
        last_log_time = time.time()
        current_chunk = []
        
        async def handle_event(event: dict):
            nonlocal last_log_time
            """Collect response events from Pi Agent."""
            event_type = event.get("type", "")
            
            # Pi Agent RPC protocol event types
            if event_type == "response":
                # Command acknowledgment
                if not event.get("success", False):
                    error_msg = event.get("error", "Unknown error")
                    response_text.append(f"[Error: {error_msg}]")
                    response_complete.set()
            
            elif event_type == "message_update":
                # Streaming text update
                data = event.get("assistantMessageEvent", {})
                if data.get("type") == "text_delta":
                    text = data.get("delta", "")
                    if text:
                        response_text.append(text)
                        current_chunk.append(text)
                        print(text, end="", flush=True)
                        
                        # Log every 5 seconds if we have data
                        now = time.time()
                        if now - last_log_time > 5.0 and current_chunk:
                            logger.info(f"[Pi Stream] {''.join(current_chunk)}")
                            current_chunk.clear()
                            last_log_time = now
            
            elif event_type == "text_delta":
                # Legacy or direct text delta
                text = event.get("text", "")
                if text:
                    response_text.append(text)
                    current_chunk.append(text)
                    print(text, end="", flush=True)
                    
                    now = time.time()
                    if now - last_log_time > 5.0 and current_chunk:
                        logger.info(f"[Pi Stream] {''.join(current_chunk)}")
                        current_chunk.clear()
                        last_log_time = now
            
            elif event_type == "message":
                # Complete message
                content = event.get("content", "")
                if content:
                    response_text.append(content)
                    print(content, end="", flush=True)
                    logger.info(f"[Pi Response Chunk] {content[:200]}...")
            
            elif event_type == "agent_end":
                if current_chunk:
                    logger.info(f"[Pi Stream Final] {''.join(current_chunk)}")
                print("\n[Done]") # New line after streaming
                response_complete.set()
            
            elif event_type in ("done", "end"):
                response_complete.set()
            
            elif event_type == "error":
                error_msg = event.get("message", event.get("error", "Unknown error"))
                logger.error(f"[ask_pi_agent] Error: {error_msg}")
                response_text.append(f"[Error: {error_msg}]")
                response_complete.set()
        
        # Set up callback
        bridge.set_callback(handle_event, callback_id=callback_id)
        
        try:
            # Ensure bridge is started
            if not bridge.running:
                await bridge.start()
                await asyncio.sleep(2.0)
            
            # Send the prompt
            await bridge.send_prompt(prompt)
            
            # Wait for response with timeout
            try:
                await asyncio.wait_for(response_complete.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[ask_pi_agent] Timeout after {timeout}s")
                response_text.append(f"[Timeout: No response within {timeout}s]")
            
            result = "".join(response_text)
            logger.info(f"[ask_pi_agent] Response: {result[:100]}...")
            
            return result
        finally:
            # Clean up callback
            bridge.remove_callback(callback_id)
    
    tools["ask_pi_agent"] = ask_pi_agent
    
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
