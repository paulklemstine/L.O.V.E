import os
import functools
from typing import Optional, Any, Dict
from langsmith import traceable, Client

# Initialize client to ensure connection early (lazy load if needed)
_client = None

def get_client():
    global _client
    if _client is None:
        try:
            _client = Client()
        except Exception as e:
            # Fallback or silent failure if not configured, though user should configure it.
            # print(f"Warning: LangSmith client could not be initialized: {e}")
            pass
    return _client

def init_tracing(project_name: str = None):
    """
    Initializes tracing configuration.
    Generally handled by env vars, but can set project name here.
    """
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    
    # Ensure tracing is on
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

def log_feedback(run_id: str, key: str, score: float, comment: str = None, correction: Dict = None):
    """
    Logs user feedback to a specific run.
    """
    client = get_client()
    if not client or not run_id:
        return
    
    try:
        client.create_feedback(
            run_id,
            key=key,
            score=score,
            comment=comment,
            correction=correction
        )
    except Exception as e:
        # Avoid crashing specifically on feedback logging
        print(f"Error logging feedback to LangSmith: {e}")

# Re-export traceable for convenience so other modules import from here
traceable = traceable
