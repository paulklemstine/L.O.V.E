import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Task:
    """Represents a unit of work for the autonomous system."""
    description: str
    source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority_score: Optional[float] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.description:
            raise ValueError("Task description cannot be empty.")
        if not self.source:
            raise ValueError("Task source cannot be empty.")

    def to_dict(self):
        """Converts the task object to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "priority_score": self.priority_score,
            "status": self.status,
            "created_at": self.created_at,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a task object from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            description=data.get("description"),
            source=data.get("source"),
            priority_score=data.get("priority_score"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            dependencies=data.get("dependencies", []),
        )
