import json
import os
from typing import List, Dict, Any, Optional
import uuid

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
    # Ensure each story has a unique ID for manipulation.
    stories_with_ids = []
    for story in user_stories:
        if "id" not in story:
            story["id"] = str(uuid.uuid4())
        stories_with_ids.append(story)

    state = {
        "user_stories": stories_with_ids,
        "current_story_index": 0,
        "active": True,
        "current_task_id": None # Reset task ID when setting new stories
    }
    save_evolution_state(state)


def get_user_stories() -> List[Dict[str, Any]]:
    """Gets all user stories."""
    state = load_evolution_state()
    return state.get("user_stories", [])


def add_user_story(title: str, description: str) -> Dict[str, Any]:
    """Adds a new user story to the backlog."""
    state = load_evolution_state()
    new_story = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
    }
    state["user_stories"].append(new_story)
    save_evolution_state(state)
    return new_story


def remove_user_story(story_id: str) -> bool:
    """Removes a user story from the backlog by its ID."""
    state = load_evolution_state()
    original_length = len(state["user_stories"])
    state["user_stories"] = [s for s in state["user_stories"] if s.get("id") != story_id]
    if len(state["user_stories"]) < original_length:
        # If the removed story was the current one, advance the index.
        # This is a simple approach; a more robust solution might be needed
        # if stories can be removed while the evolution process is active.
        if state["active"] and state["current_story_index"] >= len(state["user_stories"]):
            state["current_story_index"] = len(state["user_stories"]) - 1
            if state["current_story_index"] < 0:
                clear_evolution_state()

        save_evolution_state(state)
        return True
    return False


def reorder_user_stories(story_ids: List[str]) -> bool:
    """Reorders the user stories based on a provided list of IDs."""
    state = load_evolution_state()
    story_map = {s["id"]: s for s in state["user_stories"]}

    if len(story_ids) != len(state["user_stories"]) or set(story_ids) != set(story_map.keys()):
        return False

    reordered_stories = [story_map[id] for id in story_ids]
    state["user_stories"] = reordered_stories
    save_evolution_state(state)
    return True

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
