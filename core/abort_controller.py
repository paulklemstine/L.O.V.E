"""
Story 6.1: The "Override" Button

Provides a global abort mechanism for halting agent loops
and dangerous operations immediately.
"""
import threading
import signal
from datetime import datetime
from typing import Optional, Callable
from core.logging import log_event


# Global abort signal - thread-safe
_STOP_FLAG = threading.Event()

# Abort metadata
_abort_reason = ""
_abort_timestamp = None
_abort_callbacks = []


def request_abort(reason: str = "User requested abort") -> None:
    """Sets the global abort flag to stop all agent operations."""
    global _abort_reason, _abort_timestamp
    
    _abort_reason = reason
    _abort_timestamp = datetime.now()
    _STOP_FLAG.set()
    
    log_event(f"🛑 ABORT REQUESTED: {reason}", "WARNING")
    
    for callback in _abort_callbacks:
        try:
            callback(reason)
        except Exception as e:
            log_event(f"Abort callback error: {e}", "ERROR")


def check_abort() -> bool:
    """Checks if an abort has been requested."""
    return _STOP_FLAG.is_set()


def acknowledge_abort() -> dict:
    """Acknowledges the abort and returns metadata."""
    if not _STOP_FLAG.is_set():
        return {"aborted": False}
    
    log_event("✓ Abort acknowledged by agent", "INFO")
    return {
        "aborted": True,
        "reason": _abort_reason,
        "timestamp": _abort_timestamp.isoformat() if _abort_timestamp else None,
    }


def reset_abort() -> None:
    """Resets the abort flag for next run."""
    global _abort_reason, _abort_timestamp
    
    _STOP_FLAG.clear()
    _abort_reason = ""
    _abort_timestamp = None
    log_event("Abort flag reset", "DEBUG")


def register_abort_callback(callback: Callable[[str], None]) -> None:
    """Registers a callback for abort events."""
    _abort_callbacks.append(callback)


def get_abort_status() -> dict:
    """Gets full abort status."""
    return {
        "is_aborted": _STOP_FLAG.is_set(),
        "reason": _abort_reason,
        "timestamp": _abort_timestamp.isoformat() if _abort_timestamp else None,
    }


def setup_signal_handlers() -> None:
    """Sets up SIGTERM and SIGINT handlers."""
    def handler(signum, frame):
        names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}
        request_abort(f"Received {names.get(signum, signum)}")
    
    try:
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
    except Exception as e:
        log_event(f"Could not set signal handlers: {e}", "WARNING")


class AbortableOperation:
    """Context manager for abortable operations."""
    
    def __init__(self, name: str):
        self.name = name
        self._aborted = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return False
    
    def should_stop(self) -> bool:
        if check_abort():
            self._aborted = True
            return True
        return False


class AbortException(Exception):
    """Exception raised when abort is requested."""
    pass


def abortable(func):
    """Decorator for abortable functions."""
    def wrapper(*args, **kwargs):
        if check_abort():
            log_event(f"Skipping {func.__name__} due to abort", "WARNING")
            return None
        return func(*args, **kwargs)
    return wrapper
