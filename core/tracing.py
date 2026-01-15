"""
Tracing module - DISABLED.
LangSmith integration has been removed as it provides no value and creates
excessive API overhead (connection every ~15 seconds).

This module provides no-op stubs to maintain API compatibility.
"""
import os
import functools
from typing import Optional, Any, Dict, Callable


def get_client():
    """No-op: LangSmith client disabled."""
    return None


def init_tracing(project_name: str = None):
    """
    No-op: LangSmith tracing is disabled.
    Explicitly disables tracing to prevent any background activity.
    """
    # Explicitly disable LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGSMITH_API_KEY", None)  # Remove API key to prevent any calls


def log_feedback(run_id: str, key: str, score: float, comment: str = None, correction: Dict = None):
    """No-op: LangSmith feedback logging disabled."""
    pass


def traceable(
    run_type: str = None,
    name: str = None,
    metadata: dict = None,
    tags: list = None,
    **kwargs
) -> Callable:
    """
    No-op decorator that replaces @traceable from langsmith.
    Simply returns the original function unchanged.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
