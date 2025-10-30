import json
import os
from typing import List, Dict, Any, Optional

DESIRE_STATE_FILE = "desires.json"

def get_desire_state_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), DESIRE_STATE_FILE)

def load_desire_state() -> Dict[str, Any]:
    """Loads the desire state from the JSON file."""
    path = get_desire_state_path()
    default_state = {
        "desires": [],
        "current_desire_index": -1,
        "active": False,
        "current_task_id": None
    }
    if not os.path.exists(path):
        return default_state
    try:
        with open(path, 'r') as f:
            state = json.load(f)
            state.setdefault("current_task_id", None) # Ensure compatibility
            return state
    except (json.JSONDecodeError, IOError):
        return default_state

def save_desire_state(state: Dict[str, Any]) -> None:
    """Saves the desire state to the JSON file."""
    path = get_desire_state_path()
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

def set_desires(desires: List[Dict[str, str]]) -> None:
    """Sets the desires and activates the process."""
    state = {
        "desires": desires,
        "current_desire_index": 0,
        "active": True,
        "current_task_id": None
    }
    save_desire_state(state)

def get_current_desire() -> Optional[Dict[str, str]]:
    """Gets the current desire to be implemented."""
    state = load_desire_state()
    if not state["active"] or not state["desires"]:
        return None
    index = state["current_desire_index"]
    if 0 <= index < len(state["desires"]):
        return state["desires"][index]
    return None

def set_current_task_id_for_desire(task_id: str) -> None:
    """Sets the task ID for the current desire."""
    state = load_desire_state()
    if state["active"]:
        state["current_task_id"] = task_id
        save_desire_state(state)

def advance_to_next_desire() -> None:
    """Marks the current desire as complete and advances to the next one."""
    state = load_desire_state()
    if state["active"]:
        state["current_desire_index"] += 1
        state["current_task_id"] = None  # Reset for the next desire
        if state["current_desire_index"] >= len(state["desires"]):
            clear_desire_state()
        else:
            save_desire_state(state)

def clear_desire_state() -> None:
    """Clears the desire state, stopping the process."""
    state = {
        "desires": [],
        "current_desire_index": -1,
        "active": False,
        "current_task_id": None
    }
    save_desire_state(state)
