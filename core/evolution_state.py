"""
Evolution State Management

Manages the persistent state of the agent's evolution, including user stories,
current tasks, and tool specifications.

Migrated from lovev1 and enhanced for Epic 1.
"""

import json
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

EVOLUTION_STATE_FILE = "evolution_state.json"


@dataclass
class EvolutionarySpecification:
    """
    Specification for a tool that needs to be fabricated.
    
    Story 1.1: Generated when a tool gap is detected, contains all
    information needed for the ToolFabricator to generate code.
    """
    functional_name: str
    required_arguments: Dict[str, str]  # name -> type string
    expected_output: str
    safety_constraints: List[str] = field(default_factory=list)
    trigger_context: str = ""  # What caused the gap detection
    priority: int = 3  # 1-5, higher = more urgent
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, fabricating, validating, active, failed
    id: str = field(default_factory=lambda: str(hash(datetime.now().isoformat()))[-8:])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionarySpecification':
        return cls(**data)


def get_evolution_state_path() -> str:
    """Get absolute path to evolution state file (in project root)."""
    # .../core/evolution_state.py -> .../core -> .../L.O.V.E
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), EVOLUTION_STATE_FILE)


def load_evolution_state() -> Dict[str, Any]:
    """Loads the evolution state from the JSON file."""
    path = get_evolution_state_path()
    default_state = {
        "user_stories": [],
        "current_story_index": -1,
        "active": False,
        "current_task_id": None,
        "tool_specifications": [] # Epic 1 enhancement
    }
    
    if not os.path.exists(path):
        return default_state
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            # Ensure compatibility
            state.setdefault("current_task_id", None)
            state.setdefault("tool_specifications", [])
            return state
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load evolution state: {e}")
        return default_state


def save_evolution_state(state: Dict[str, Any]) -> None:
    """Saves the evolution state to the JSON file."""
    path = get_evolution_state_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save evolution state: {e}")


# =============================================================================
# User Story Management
# =============================================================================

def set_user_stories(user_stories: List[Dict[str, str]]) -> None:
    """Sets the user stories and activates the evolution process."""
    stories_with_ids = []
    for story in user_stories:
        if "id" not in story:
            story["id"] = str(uuid.uuid4())
        stories_with_ids.append(story)

    state = load_evolution_state()
    state["user_stories"] = stories_with_ids
    state["current_story_index"] = 0
    state["active"] = True
    state["current_task_id"] = None
    
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
    original_stories = state["user_stories"]
    state["user_stories"] = [s for s in original_stories if s.get("id") != story_id]
    
    if len(state["user_stories"]) < len(original_stories):
        # Adjust index if necessary
        if state["active"] and state["current_story_index"] >= len(state["user_stories"]):
             state["current_story_index"] = max(0, len(state["user_stories"]) - 1)
             if not state["user_stories"]:
                 state["active"] = False
                 
        save_evolution_state(state)
        return True
    return False


def get_current_story() -> Optional[Dict[str, str]]:
    """Gets the current user story to be implemented."""
    state = load_evolution_state()
    if not state["active"] or not state["user_stories"]:
        return None
        
    index = state["current_story_index"]
    if 0 <= index < len(state["user_stories"]):
        return state["user_stories"][index]
    return None


def advance_to_next_story() -> None:
    """Marks the current story as complete and advances to the next one."""
    state = load_evolution_state()
    if state["active"]:
        state["current_story_index"] += 1
        state["current_task_id"] = None
        
        if state["current_story_index"] >= len(state["user_stories"]):
            state["active"] = False # Done
            
        save_evolution_state(state)


# =============================================================================
# Tool Specification Management (Epic 1)
# =============================================================================

def add_tool_specification(spec: EvolutionarySpecification) -> None:
    """Adds a tool specification to the backlog."""
    state = load_evolution_state()
    state["tool_specifications"].append(spec.to_dict())
    save_evolution_state(state)


def get_pending_specifications() -> List[EvolutionarySpecification]:
    """Gets all pending tool specifications."""
    state = load_evolution_state()
    specs = []
    for spec_data in state.get("tool_specifications", []):
        if spec_data.get("status") == "pending":
            specs.append(EvolutionarySpecification.from_dict(spec_data))
    return specs


def update_specification_status(spec_id: str, status: str) -> bool:
    """Updates the status of a tool specification."""
    state = load_evolution_state()
    updated = False
    
    for spec in state.get("tool_specifications", []):
        if spec.get("id") == spec_id:
            spec["status"] = status
            updated = True
            break
            
    if updated:
        save_evolution_state(state)
        
    return updated


def get_specification(spec_id: str) -> Optional[EvolutionarySpecification]:
    """Get a specific specification by ID."""
    state = load_evolution_state()
    for spec_data in state.get("tool_specifications", []):
        if spec_data.get("id") == spec_id:
            return EvolutionarySpecification.from_dict(spec_data)
    return None
