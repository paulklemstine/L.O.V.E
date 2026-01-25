import json
import os
import copy
from typing import List, Dict, Any, Optional
import uuid

DESIRE_STATE_FILE = "desire_state.json"

# In-memory cache
_cache: Optional[Dict[str, Any]] = None
_last_mtime: float = 0

def get_desire_state_path() -> str:
    """Gets the absolute path to the desire_state.json file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), DESIRE_STATE_FILE)

def load_desire_state() -> Dict[str, Any]:
    """Loads the desire state from the JSON file with caching."""
    global _cache, _last_mtime
    path = get_desire_state_path()
    default_state = {
        "desires": [],
        "current_desire_index": -1,
        "active": False,
        "current_task_id": None
    }

    try:
        current_mtime = os.path.getmtime(path)
    except OSError:
        # File likely doesn't exist
        _cache = copy.deepcopy(default_state)
        _last_mtime = 0
        return copy.deepcopy(_cache)

    # Check if cache is valid
    if _cache is not None and current_mtime == _last_mtime:
        return copy.deepcopy(_cache)

    try:
        with open(path, 'r') as f:
            state = json.load(f)
            state.setdefault("current_task_id", None)
            _cache = state
            _last_mtime = current_mtime
            return copy.deepcopy(_cache)
    except (json.JSONDecodeError, IOError):
        # Fallback to default state on error
        return default_state

def save_desire_state(state: Dict[str, Any]) -> None:
    """Saves the desire state to the JSON file and updates cache."""
    global _cache, _last_mtime
    path = get_desire_state_path()
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

    # Update cache
    _cache = copy.deepcopy(state)
    try:
        _last_mtime = os.path.getmtime(path)
    except OSError:
        _last_mtime = 0

def set_desires(desires: List[Dict[str, Any]]) -> None:
    """Sets the desires and activates the process."""
    desires_with_ids = []
    for desire in desires:
        if "id" not in desire:
            desire["id"] = str(uuid.uuid4())
        desires_with_ids.append(desire)

    state = {
        "desires": desires_with_ids,
        "current_desire_index": 0,
        "active": True,
        "current_task_id": None
    }
    save_desire_state(state)

def get_current_desire() -> Optional[Dict[str, Any]]:
    """Gets the current desire to be implemented."""
    state = load_desire_state()
    if not state.get("active") or not state.get("desires"):
        return None
    index = state.get("current_desire_index", -1)
    desires = state.get("desires", [])
    if 0 <= index < len(desires):
        return desires[index]
    return None

def set_current_task_id_for_desire(task_id: str) -> None:
    """Sets the task ID for the current desire."""
    state = load_desire_state()
    if state.get("active"):
        state["current_task_id"] = task_id
        save_desire_state(state)

def advance_to_next_desire() -> None:
    """Marks the current desire as complete and advances to the next one."""
    state = load_desire_state()
    if state.get("active"):
        state["current_desire_index"] += 1
        state["current_task_id"] = None
        if state["current_desire_index"] >= len(state.get("desires", [])):
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

# --- Functions for God Agent Tools ---

def get_desires() -> List[Dict[str, Any]]:
    """God Agent Tool: Gets all desires."""
    state = load_desire_state()
    return state.get("desires", [])

def add_desire(title: str, description: str, plan: List[str] = None, notes: str = "") -> Dict[str, Any]:
    """God Agent Tool: Adds a new desire to the backlog."""
    state = load_desire_state()
    new_desire = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "plan": plan if plan is not None else [],
        "notes": notes,
        "state": "pending"
    }
    state["desires"].append(new_desire)
    save_desire_state(state)
    return new_desire

def remove_desire(desire_id: str) -> bool:
    """God Agent Tool: Removes a desire from the backlog by its ID."""
    state = load_desire_state()
    original_length = len(state["desires"])
    state["desires"] = [d for d in state["desires"] if d.get("id") != desire_id]
    if len(state["desires"]) < original_length:
        save_desire_state(state)
        return True
    return False

def reorder_desires(desire_ids: List[str]) -> bool:
    """God Agent Tool: Reorders the desires based on a provided list of IDs."""
    state = load_desire_state()
    desire_map = {d["id"]: d for d in state["desires"]}

    if len(desire_ids) != len(state["desires"]) or set(desire_ids) != set(desire_map.keys()):
        return False

    reordered_desires = [desire_map[id] for id in desire_ids]
    state["desires"] = reordered_desires
    save_desire_state(state)
    return True
