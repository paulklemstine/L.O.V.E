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
            "last_update": self.last_update.isoformat(),
            "log_count": len(self.logs)
        }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent logs."""
        return list(self.logs)[-limit:]

# Global accessor
def get_state_manager() -> StateManager:
    return StateManager()
