import asyncio
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.logs = deque(maxlen=1000)
        self.current_goal: Optional[str] = None
        self.iteration: int = 0
        self.memory_stats: Dict[str, Any] = {}
        self.is_running: bool = False
        self.agent_states: Dict[str, Dict[str, str]] = {}  # {agent_name: {status, action, thought}}
        self.last_image: Optional[str] = None  # Base64 string
        self.latest_post: Optional[Dict[str, Any]] = None
        self.interactions: List[Dict[str, Any]] = []
        self.last_update = datetime.now()
        self._initialized = True

    def add_log(self, level: str, message: str, source: str = "System"):
        """Add a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "source": source
        }
        self.logs.append(entry)
        self.last_update = datetime.now()

    def update_agent_status(self, agent_name: str, status: str, action: str = None, thought: str = None):
        """Update the status of a specific agent."""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = {}
        
        self.agent_states[agent_name]["status"] = status
        if action:
            self.agent_states[agent_name]["action"] = action
        if thought:
            self.agent_states[agent_name]["thought"] = thought
            
        self.last_update = datetime.now()

    def update_image(self, image_b64: str):
        """Update the latest generated image."""
        self.last_image = image_b64
        self.last_update = datetime.now()

    def update_latest_post(self, post: Dict[str, Any]):
        """Update the latest own post."""
        self.latest_post = post
        self.last_update = datetime.now()

    def update_interactions(self, interactions: List[Dict[str, Any]]):
        """Update recent interactions."""
        self.interactions = interactions
        self.last_update = datetime.now()

    def update_state(self, **kwargs):
        """Update arbitrary state variables."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now()

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current state."""
        return {
            "iteration": self.iteration,
            "current_goal": self.current_goal,
            "is_running": self.is_running,
            "memory_stats": self.memory_stats,
            "agent_states": self.agent_states,
            "last_image": self.last_image,
            "latest_post": self.latest_post,
            "interactions": self.interactions,
            "last_update": self.last_update.isoformat(),
            "log_count": len(self.logs)
        }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent logs."""
        return list(self.logs)[-limit:]

# Global accessor
def get_state_manager() -> StateManager:
    return StateManager()
