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
        self.chat_history: List[Dict[str, Any]] = []
        self.command_queue: deque = deque()
        self.current_command: Optional[str] = None
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

    def update_agent_status(self, agent_name: str, status: str, action: str = None, thought: str = None, subtasks: List[str] = None, info: Dict[str, Any] = None):
        """Update the status of a specific agent."""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = {}
        
        self.agent_states[agent_name]["status"] = status
        if action:
            self.agent_states[agent_name]["action"] = action
        if thought:
            self.agent_states[agent_name]["thought"] = thought
        if subtasks is not None:
             self.agent_states[agent_name]["subtasks"] = subtasks
        if info is not None:
             self.agent_states[agent_name]["info"] = info
            
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
            "iteration": self.iteration,
            "current_goal": self.current_goal,
            "current_command": self.current_command,
            "command_queue_len": len(self.command_queue),
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

    def add_chat_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep history manageable
        if len(self.chat_history) > 100:
            self.chat_history.pop(0)

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the full chat history."""
        return self.chat_history

    def add_command(self, command_text: str):
        """Add a command to the queue."""
        self.command_queue.append(command_text)
        self.last_update = datetime.now()

    def get_next_command(self) -> Optional[str]:
        """Get the next command from the queue, or None."""
        if self.command_queue:
            cmd = self.command_queue.popleft()
            self.current_command = cmd
            return cmd
        return None

    def clear_current_command(self):
        """Clear the currently executing command."""
        self.current_command = None
        self.last_update = datetime.now()

# Global accessor
def get_state_manager() -> StateManager:
    return StateManager()
