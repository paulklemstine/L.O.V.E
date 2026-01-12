"""
Hooks System for L.O.V.E. External Integrations.

This module provides an event-driven hooks system that allows external
sessions and integrations to subscribe to L.O.V.E. events.

Hook Types:
    - on_tool_call: Fired when any tool is executed
    - on_state_change: Fired when love_state is updated
    - on_command_received: Fired when a command is submitted
    - on_output: Fired when console output is generated
    - on_agent_step: Fired on each agent reasoning step
"""

import asyncio
import logging
import httpx
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Supported hook event types."""
    ON_TOOL_CALL = "on_tool_call"
    ON_STATE_CHANGE = "on_state_change"
    ON_COMMAND_RECEIVED = "on_command_received"
    ON_OUTPUT = "on_output"
    ON_AGENT_STEP = "on_agent_step"


@dataclass
class HookEvent:
    """Represents an event that can trigger hooks."""
    hook_type: HookType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "hook_type": self.hook_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source
        }


@dataclass
class RemoteHook:
    """A remote HTTP callback hook."""
    hook_id: str
    hook_type: HookType
    callback_url: str
    headers: Dict[str, str] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    failure_count: int = 0
    max_failures: int = 5  # Auto-disable after this many failures


@dataclass 
class LocalHook:
    """An in-process callback hook."""
    hook_id: str
    hook_type: HookType
    callback: Callable[[HookEvent], Any]
    is_async: bool = False
    active: bool = True


class HooksManager:
    """
    Manages event hooks for external integrations.
    
    Supports both remote HTTP callbacks and local in-process callbacks.
    Thread-safe and async-compatible.
    
    Example:
        manager = HooksManager()
        
        # Register a remote webhook
        manager.register_remote(
            HookType.ON_TOOL_CALL,
            "https://example.com/webhook",
            {"Authorization": "Bearer token"}
        )
        
        # Register a local callback
        def my_handler(event: HookEvent):
            print(f"Tool called: {event.data}")
        
        manager.register_local(HookType.ON_TOOL_CALL, my_handler)
        
        # Fire an event
        await manager.fire(HookType.ON_TOOL_CALL, {
            "tool_name": "search",
            "args": {"query": "test"},
            "result": "success"
        })
    """
    
    def __init__(self):
        self._remote_hooks: Dict[HookType, List[RemoteHook]] = {
            hook_type: [] for hook_type in HookType
        }
        self._local_hooks: Dict[HookType, List[LocalHook]] = {
            hook_type: [] for hook_type in HookType
        }
        self._hook_counter = 0
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    def _get_next_id(self) -> str:
        """Generate unique hook ID."""
        self._hook_counter += 1
        return f"hook_{self._hook_counter}"
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client
    
    def register_remote(
        self,
        hook_type: HookType,
        callback_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a remote HTTP webhook.
        
        Args:
            hook_type: The type of event to listen for
            callback_url: URL to POST event data to
            headers: Optional headers to include in requests
            
        Returns:
            The hook ID for later reference
        """
        hook_id = self._get_next_id()
        hook = RemoteHook(
            hook_id=hook_id,
            hook_type=hook_type,
            callback_url=callback_url,
            headers=headers or {}
        )
        self._remote_hooks[hook_type].append(hook)
        logger.info(f"Registered remote hook {hook_id} for {hook_type.value} -> {callback_url}")
        return hook_id
    
    def register_local(
        self,
        hook_type: HookType,
        callback: Callable[[HookEvent], Any],
        is_async: bool = False
    ) -> str:
        """
        Register a local in-process callback.
        
        Args:
            hook_type: The type of event to listen for
            callback: Function to call with HookEvent
            is_async: Whether the callback is async
            
        Returns:
            The hook ID for later reference
        """
        hook_id = self._get_next_id()
        hook = LocalHook(
            hook_id=hook_id,
            hook_type=hook_type,
            callback=callback,
            is_async=is_async
        )
        self._local_hooks[hook_type].append(hook)
        logger.info(f"Registered local hook {hook_id} for {hook_type.value}")
        return hook_id
    
    def unregister(self, hook_id: str) -> bool:
        """
        Unregister a hook by ID.
        
        Args:
            hook_id: The hook ID to remove
            
        Returns:
            True if found and removed, False otherwise
        """
        for hook_type in HookType:
            # Check remote hooks
            for i, hook in enumerate(self._remote_hooks[hook_type]):
                if hook.hook_id == hook_id:
                    self._remote_hooks[hook_type].pop(i)
                    logger.info(f"Unregistered remote hook {hook_id}")
                    return True
            
            # Check local hooks
            for i, hook in enumerate(self._local_hooks[hook_type]):
                if hook.hook_id == hook_id:
                    self._local_hooks[hook_type].pop(i)
                    logger.info(f"Unregistered local hook {hook_id}")
                    return True
        
        return False
    
    async def fire(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        source: str = "system"
    ) -> Dict[str, Any]:
        """
        Fire an event to all registered hooks of the given type.
        
        Args:
            hook_type: The type of event
            data: Event data payload
            source: Source identifier for the event
            
        Returns:
            Summary of hook execution results
        """
        event = HookEvent(
            hook_type=hook_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        
        results = {
            "hook_type": hook_type.value,
            "remote_success": 0,
            "remote_failed": 0,
            "local_success": 0,
            "local_failed": 0,
            "errors": []
        }
        
        # Fire remote hooks (don't block on failures)
        for hook in self._remote_hooks[hook_type]:
            if not hook.active:
                continue
            
            try:
                client = await self._get_http_client()
                response = await client.post(
                    hook.callback_url,
                    json=event.to_dict(),
                    headers=hook.headers
                )
                if response.status_code < 400:
                    results["remote_success"] += 1
                    hook.failure_count = 0
                else:
                    results["remote_failed"] += 1
                    hook.failure_count += 1
                    results["errors"].append(f"{hook.hook_id}: HTTP {response.status_code}")
            except Exception as e:
                results["remote_failed"] += 1
                hook.failure_count += 1
                results["errors"].append(f"{hook.hook_id}: {str(e)}")
            
            # Auto-disable after too many failures
            if hook.failure_count >= hook.max_failures:
                hook.active = False
                logger.warning(f"Hook {hook.hook_id} disabled after {hook.failure_count} failures")
        
        # Fire local hooks
        for hook in self._local_hooks[hook_type]:
            if not hook.active:
                continue
            
            try:
                if hook.is_async:
                    await hook.callback(event)
                else:
                    hook.callback(event)
                results["local_success"] += 1
            except Exception as e:
                results["local_failed"] += 1
                results["errors"].append(f"{hook.hook_id}: {str(e)}")
                logger.error(f"Local hook {hook.hook_id} failed: {e}")
        
        return results
    
    def list_hooks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all registered hooks.
        
        Returns:
            Dictionary of hook type -> list of hook info
        """
        result = {}
        
        for hook_type in HookType:
            hooks = []
            
            for hook in self._remote_hooks[hook_type]:
                hooks.append({
                    "hook_id": hook.hook_id,
                    "type": "remote",
                    "callback_url": hook.callback_url,
                    "active": hook.active,
                    "failure_count": hook.failure_count,
                    "created_at": hook.created_at.isoformat()
                })
            
            for hook in self._local_hooks[hook_type]:
                hooks.append({
                    "hook_id": hook.hook_id,
                    "type": "local",
                    "callback": hook.callback.__name__,
                    "is_async": hook.is_async,
                    "active": hook.active
                })
            
            if hooks:
                result[hook_type.value] = hooks
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hook system statistics."""
        total_remote = sum(len(hooks) for hooks in self._remote_hooks.values())
        total_local = sum(len(hooks) for hooks in self._local_hooks.values())
        active_remote = sum(
            sum(1 for h in hooks if h.active)
            for hooks in self._remote_hooks.values()
        )
        active_local = sum(
            sum(1 for h in hooks if h.active)
            for hooks in self._local_hooks.values()
        )
        
        return {
            "total_hooks": total_remote + total_local,
            "remote_hooks": total_remote,
            "local_hooks": total_local,
            "active_remote": active_remote,
            "active_local": active_local,
            "hook_types": [ht.value for ht in HookType]
        }
    
    async def close(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Singleton instance
_hooks_manager: Optional[HooksManager] = None


def get_hooks_manager() -> HooksManager:
    """Get the global HooksManager singleton."""
    global _hooks_manager
    if _hooks_manager is None:
        _hooks_manager = HooksManager()
    return _hooks_manager


# Convenience functions for firing common events
async def fire_tool_call(tool_name: str, args: Dict[str, Any], result: Any) -> None:
    """Fire a tool call hook event."""
    manager = get_hooks_manager()
    await manager.fire(HookType.ON_TOOL_CALL, {
        "tool_name": tool_name,
        "args": args,
        "result": str(result) if result else None
    })


async def fire_state_change(key: str, old_value: Any, new_value: Any) -> None:
    """Fire a state change hook event."""
    manager = get_hooks_manager()
    await manager.fire(HookType.ON_STATE_CHANGE, {
        "key": key,
        "old_value": old_value,
        "new_value": new_value
    })


async def fire_command(command: str, source: str = "user") -> None:
    """Fire a command received hook event."""
    manager = get_hooks_manager()
    await manager.fire(HookType.ON_COMMAND_RECEIVED, {
        "command": command
    }, source=source)


async def fire_output(panel_type: str, content: str, title: Optional[str] = None) -> None:
    """Fire an output hook event."""
    manager = get_hooks_manager()
    await manager.fire(HookType.ON_OUTPUT, {
        "panel_type": panel_type,
        "content": content,
        "title": title
    })


async def fire_agent_step(step_type: str, details: Dict[str, Any]) -> None:
    """Fire an agent step hook event."""
    manager = get_hooks_manager()
    await manager.fire(HookType.ON_AGENT_STEP, {
        "step_type": step_type,
        **details
    })
