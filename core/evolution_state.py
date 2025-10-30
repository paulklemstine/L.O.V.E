import json
import os
from typing import List, Dict, Any, Optional

EVOLUTION_STATE_FILE = "evolution_state.json"

def get_evolution_state_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), EVOLUTION_STATE_FILE)

def load_evolution_state() -> Dict[str, Any]:
    """Loads the evolution state from the JSON file."""
    path = get_evolution_state_path()
    default_state = {
        "user_stories": [],
        "current_story_index": -1,
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

def save_evolution_state(state: Dict[str, Any]) -> None:
    """Saves the evolution state to the JSON file."""
    path = get_evolution_state_path()
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

def set_user_stories(user_stories: List[Dict[str, str]]) -> None:
    """Sets the user stories and activates the evolution process."""
    state = {
        "user_stories": user_stories,
        "current_story_index": 0,
        "active": True
    }
    save_evolution_state(state)

def get_current_story() -> Optional[Dict[str, str]]:
    """Gets the current user story to be implemented."""
    state = load_evolution_state()
    if not state["active"] or not state["user_stories"]:
        return None
    index = state["current_story_index"]
    if 0 <= index < len(state["user_stories"]):
        return state["user_stories"][index]
    return None

def set_current_task_id(task_id: str) -> None:
    """Sets the task ID for the current user story."""
    state = load_evolution_state()
    if state["active"]:
        state["current_task_id"] = task_id
        save_evolution_state(state)

def advance_to_next_story() -> None:
    """Marks the current story as complete and advances to the next one."""
    state = load_evolution_state()
    if state["active"]:
        state["current_story_index"] += 1
        state["current_task_id"] = None  # Reset for the next story
        if state["current_story_index"] >= len(state["user_stories"]):
            clear_evolution_state()
        else:
            save_evolution_state(state)

def clear_evolution_state() -> None:
    """Clears the evolution state, stopping the process."""
    state = {
        "user_stories": [],
        "current_story_index": -1,
        "active": False
    }
    save_evolution_state(state)
