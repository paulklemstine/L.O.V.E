# core/tool_invocation_hooks.py
"""
Tool Invocation Hooks for News Feed UI Display.

Provides centralized hooks to display fancy UI elements when tools are invoked,
using the terminal widget panel from display.py.
"""

import time
from typing import Any, Callable, Dict, List, Optional
from functools import wraps

import core.logging
import core.shared_state as shared_state


class ToolInvocationDisplay:
    """
    Manages tool invocation display in the news feed.
    
    Creates terminal widget panels showing:
    - Tool name and status (thinking/executing/complete/error)
    - Arguments being passed
    - Streaming stdout/stderr output
    - Elapsed time
    """
    
    def __init__(self, ui_queue=None):
        self._ui_queue = ui_queue
        self._listeners: List[Callable] = []
    
    @property
    def ui_queue(self):
        """Get UI queue from shared_state if not provided directly."""
        if self._ui_queue:
            return self._ui_queue
        return getattr(shared_state, 'ui_panel_queue', None)
    
    def add_listener(self, callback: Callable):
        """Add a callback to be notified on tool invocation events."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable):
        """Remove a listener callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self, event: Dict[str, Any]):
        """Notify all listeners of a tool invocation event."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                core.logging.log_event(
                    f"Tool invocation listener error: {e}",
                    "WARNING"
                )
    
    def display_invocation(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        status: str = "executing",
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        elapsed_time: Optional[float] = None,
        width: int = 80
    ):
        """
        Display a tool invocation in the news feed.
        
        Args:
            tool_name: Name of the tool being invoked
            arguments: Dictionary of arguments passed to the tool
            status: One of "thinking", "executing", "complete", "error"
            stdout: Standard output from the tool (if any)
            stderr: Standard error from the tool (if any)
            elapsed_time: Time taken for execution in seconds
            width: Width of the panel (default 80)
        """
        queue = self.ui_queue
        if not queue:
            return
        
        try:
            # Import here to avoid circular imports
            from display import create_terminal_widget_panel
            
            panel = create_terminal_widget_panel(
                tool_name=tool_name,
                arguments=arguments,
                stdout=stdout,
                stderr=stderr,
                status=status,
                elapsed_time=elapsed_time,
                width=width
            )
            
            queue.put(panel)
            
            # Notify listeners
            self._notify_listeners({
                "tool_name": tool_name,
                "arguments": arguments,
                "status": status,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_time": elapsed_time
            })
            
        except Exception as e:
            core.logging.log_event(
                f"Error displaying tool invocation: {e}",
                "WARNING"
            )
    
    def display_start(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ):
        """Display tool invocation start."""
        self.display_invocation(
            tool_name=tool_name,
            arguments=arguments,
            status="executing"
        )
    
    def display_complete(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
        elapsed_time: Optional[float] = None
    ):
        """Display tool invocation completion."""
        self.display_invocation(
            tool_name=tool_name,
            arguments=arguments,
            status="complete",
            stdout=result,
            elapsed_time=elapsed_time
        )
    
    def display_error(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        elapsed_time: Optional[float] = None
    ):
        """Display tool invocation error."""
        self.display_invocation(
            tool_name=tool_name,
            arguments=arguments,
            status="error",
            stderr=error,
            elapsed_time=elapsed_time
        )


# Global instance
_display_instance: Optional[ToolInvocationDisplay] = None


def get_tool_display() -> ToolInvocationDisplay:
    """Get or create the global ToolInvocationDisplay instance."""
    global _display_instance
    if _display_instance is None:
        _display_instance = ToolInvocationDisplay()
    return _display_instance


def display_tool_invocation(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    status: str = "executing",
    result: Optional[str] = None,
    error: Optional[str] = None,
    elapsed_time: Optional[float] = None
):
    """
    Convenience function to display a tool invocation in the news feed.
    
    Args:
        tool_name: Name of the tool
        arguments: Arguments dict
        status: Status string (executing, complete, error)
        result: Result output (for stdout)
        error: Error output (for stderr)
        elapsed_time: Elapsed time in seconds
    """
    display = get_tool_display()
    
    if status == "complete":
        display.display_complete(tool_name, arguments, result, elapsed_time)
    elif status == "error":
        display.display_error(tool_name, arguments, error, elapsed_time)
    else:
        display.display_start(tool_name, arguments)


def with_tool_display(func: Callable) -> Callable:
    """
    Decorator that automatically displays tool invocation in the news feed.
    
    Wraps a tool function to show:
    - "executing" status when called
    - "complete" status with result on success
    - "error" status on exception
    
    Works with both sync and async functions.
    """
    import asyncio
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = getattr(func, 'name', func.__name__)
        display = get_tool_display()
        start_time = time.time()
        
        # Show executing status
        display.display_start(tool_name, kwargs)
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Truncate result for display
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            
            display.display_complete(tool_name, kwargs, result_str, elapsed)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            display.display_error(tool_name, kwargs, str(e), elapsed)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        tool_name = getattr(func, 'name', func.__name__)
        display = get_tool_display()
        start_time = time.time()
        
        # Show executing status
        display.display_start(tool_name, kwargs)
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Truncate result for display
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            
            display.display_complete(tool_name, kwargs, result_str, elapsed)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            display.display_error(tool_name, kwargs, str(e), elapsed)
            raise
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
