"""Base class for all tools with UI visibility hooks."""
import time
from typing import Any, Optional
import asyncio


class ToolBase:
    """Base class for all tools with pre/post execution hooks for UI visibility."""
    
    name = "tool_base"
    description = "This is a base class and should not be used directly."
    ui_queue = None  # Set by engine at tool registration
    
    def __init__(self):
        self._start_time: Optional[float] = None
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Main entry point. Wraps execute() with visibility notifications.
        """
        self._notify_start(args, kwargs)
        try:
            result = self.execute(*args, **kwargs)
            self._notify_complete(result)
            return result
        except Exception as e:
            self._notify_error(e)
            raise
    
    def execute(self, *args, **kwargs) -> Any:
        """Override this method in subclasses to implement tool logic."""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def _notify_start(self, args: tuple, kwargs: dict) -> None:
        """Sends a 'executing' status to the UI queue."""
        self._start_time = time.time()
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                arguments=kwargs if kwargs else dict(zip(range(len(args)), args)),
                status="executing",
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _notify_complete(self, result: Any) -> None:
        """Sends a 'complete' status to the UI queue."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            # Truncate result for display
            result_str = str(result)
            stdout = result_str[:500] if len(result_str) > 500 else result_str
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                status="complete",
                stdout=stdout,
                elapsed_time=elapsed,
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _notify_error(self, error: Exception) -> None:
        """Sends an 'error' status to the UI queue."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                status="error",
                stderr=str(error),
                elapsed_time=elapsed,
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _put_to_queue(self, item: dict) -> None:
        """Helper to put items into the UI queue, handling both sync and async queues."""
        if not self.ui_queue:
            return
        
        if isinstance(self.ui_queue, asyncio.Queue):
            try:
                self.ui_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass  # Drop if queue is full
        else:
            # Assume queue.Queue (synchronous)
            try:
                self.ui_queue.put_nowait(item)
            except:
                pass  # Drop if queue is full


class ToolWrapper:
    """
    Wraps legacy function-based tools with UI visibility hooks.
    Used for tools that aren't ToolBase subclasses.
    """
    
    def __init__(self, func, name: str, ui_queue=None):
        self.func = func
        self.name = name
        self.ui_queue = ui_queue
        self._start_time: Optional[float] = None
        # Copy function metadata
        self.__doc__ = func.__doc__
        self.__name__ = getattr(func, '__name__', name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Wraps the function call with visibility notifications."""
        self._notify_start(args, kwargs)
        try:
            result = self.func(*args, **kwargs)
            self._notify_complete(result)
            return result
        except Exception as e:
            self._notify_error(e)
            raise
    
    def _notify_start(self, args: tuple, kwargs: dict) -> None:
        """Sends an 'executing' status to the UI queue."""
        self._start_time = time.time()
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                arguments=kwargs if kwargs else dict(zip(range(len(args)), args)),
                status="executing",
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _notify_complete(self, result: Any) -> None:
        """Sends a 'complete' status to the UI queue."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            result_str = str(result)
            stdout = result_str[:500] if len(result_str) > 500 else result_str
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                status="complete",
                stdout=stdout,
                elapsed_time=elapsed,
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _notify_error(self, error: Exception) -> None:
        """Sends an 'error' status to the UI queue."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self.ui_queue:
            from display import create_terminal_widget_panel, get_terminal_width
            panel = create_terminal_widget_panel(
                tool_name=self.name,
                status="error",
                stderr=str(error),
                elapsed_time=elapsed,
                width=get_terminal_width()
            )
            self._put_to_queue({"type": "terminal_widget", "content": panel})
    
    def _put_to_queue(self, item: dict) -> None:
        """Helper to put items into the UI queue."""
        if not self.ui_queue:
            return
        
        if isinstance(self.ui_queue, asyncio.Queue):
            try:
                self.ui_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
        else:
            try:
                self.ui_queue.put_nowait(item)
            except:
                pass