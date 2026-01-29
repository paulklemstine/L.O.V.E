"""
beads.py - Persistent Atomic Task Tracking (The "Gastown" Pattern)

"Beads" are tiny, trackable units of work. They survive session crashes
and provide a persistent state for the agent's work queue.
"""

import json
import logging
import uuid
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("Beads")

class BeadState(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class Bead:
    """An atomic unit of work."""
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: BeadState = BeadState.PENDING
    assigned_worker: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    priority: int = 1  # 1 (low) to 5 (critical)
    parent_bead_id: Optional[str] = None  # For sub-tasks

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bead':
        # Handle enum conversion
        if "status" in data:
            data["status"] = BeadState(data["status"])
        return cls(**data)

    def mark_started(self, worker_id: str):
        self.status = BeadState.IN_PROGRESS
        self.assigned_worker = worker_id
        self.updated_at = time.time()

    def mark_review(self, result: str):
        self.status = BeadState.REVIEW
        self.result = result
        self.updated_at = time.time()

    def mark_complete(self):
        self.status = BeadState.COMPLETED
        self.updated_at = time.time()

    def mark_failed(self, error: str):
        self.status = BeadState.FAILED
        self.result = error
        self.updated_at = time.time()

class BeadChain:
    """
    Manages the persistent list of Beads.
    Acts as the source of truth for all work in the system.
    """
    def __init__(self, persistence_path: str = "state/beads.json"):
        self.persistence_path = Path(persistence_path)
        self.beads: Dict[str, Bead] = {}
        self._ensure_persistence_dir()
        self.load()

    def _ensure_persistence_dir(self):
        # Create parent directory if it doesn't exist
        if not self.persistence_path.parent.exists():
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

    def create_bead(self, description: str, priority: int = 1, context: Dict = None) -> Bead:
        bead = Bead(
            description=description,
            priority=priority,
            context=context or {}
        )
        self.beads[bead.id] = bead
        self.save()
        logger.info(f"Created Bead [{bead.id}]: {description}")
        return bead

    def get_bead(self, bead_id: str) -> Optional[Bead]:
        return self.beads.get(bead_id)

    def get_next_pending(self) -> Optional[Bead]:
        """Get the highest priority pending bead."""
        pending = [b for b in self.beads.values() if b.status == BeadState.PENDING]
        if not pending:
            return None
        # Sort by priority (descending) then creation time (ascending)
        pending.sort(key=lambda b: (-b.priority, b.created_at))
        return pending[0]

    def get_assigned_to(self, worker_id: str) -> List[Bead]:
        return [b for b in self.beads.values() if b.assigned_worker == worker_id and b.status == BeadState.IN_PROGRESS]

    def save(self):
        try:
            data = {bid: b.to_dict() for bid, b in self.beads.items()}
            self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save beads: {e}")

    def load(self):
        if not self.persistence_path.exists():
            return
        
        try:
            content = self.persistence_path.read_text()
            if not content.strip():
                return
                
            data = json.loads(content)
            self.beads = {bid: Bead.from_dict(b_data) for bid, b_data in data.items()}
            logger.info(f"Loaded {len(self.beads)} beads from {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to load beads: {e}")
            # Don't crash, just start empty
            self.beads = {}

    def clean_archived(self):
        """Remove archived beads to save space."""
        # Convert to list to avoid runtime error during iteration
        to_remove = [bid for bid, b in self.beads.items() if b.status == BeadState.ARCHIVED]
        for bid in to_remove:
            del self.beads[bid]
        self.save()

# Global accessor
_bead_chain = None
def get_bead_chain() -> BeadChain:
    global _bead_chain
    if _bead_chain is None:
        # Assuming run from love2 root or similar, adjust path as needed
        _bead_chain = BeadChain(persistence_path="state/beads.json")
    return _bead_chain
