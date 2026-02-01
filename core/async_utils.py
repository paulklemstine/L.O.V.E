"""
async_utils.py - Asyncio Utilities for L.O.V.E.

Provides helpers for safely running async code in environments with pre-existing
event loops (like Google Colab, Jupyter, or IPython).
"""

import asyncio
import threading
from typing import Coroutine, Any, TypeVar

T = TypeVar('T')

def run_sync_safe(coroutine: Coroutine[Any, Any, T]) -> T:
    """
    Safely run a coroutine synchronously, even if an event loop is already running.
    Useful for Colab/Jupyter compatibility where the main loop is active.
    
    Args:
        coroutine: The coroutine to execute.
        
    Returns:
        The result of the coroutine.
        
    Raises:
        Exception: Any exception raised by the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        # Loop is running, use a thread to avoid RuntimeError: 
        # "asyncio.run() cannot be called from a running event loop"
        
        result = None
        exception = None
        
        def runner():
            nonlocal result, exception
            try:
                # asyncio.run creates a new event loop for this thread
                result = asyncio.run(coroutine)
            except Exception as e:
                exception = e
                
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result # type: ignore
    else:
        # No running loop, standard execution
        return asyncio.run(coroutine)
