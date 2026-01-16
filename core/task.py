import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Task:
    """
    Represents a unit of work for the autonomous system.
    
    Enhanced with review tracking for the upgraded task decomposition system.
    """
    description: str
    source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority_score: Optional[float] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    
    # Review tracking fields
    review_status: str = "unreviewed"  # unreviewed, approved, rejected, needs_refinement, gated
    review_feedback: Optional[str] = None
    reviewer_id: Optional[str] = None
    review_confidence: float = 0.0
    
    # Meta review tracking (for plan-level review)
    meta_review_status: str = "unreviewed"  # unreviewed, approved, rejected, needs_refinement
    meta_review_feedback: Optional[str] = None
    
    # Validation tracking
    validation_status: str = "unvalidated"  # unvalidated, valid, invalid, gated
    validation_reason: Optional[str] = None
    
    # Category for routing to appropriate executors
    category: str = "general"  # general, financial, social, technical, abstract

    def __post_init__(self):
        if not self.description:
            raise ValueError("Task description cannot be empty.")
        if not self.source:
            raise ValueError("Task source cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the task object to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "priority_score": self.priority_score,
            "status": self.status,
            "created_at": self.created_at,
            "dependencies": self.dependencies,
            "review_status": self.review_status,
            "review_feedback": self.review_feedback,
            "reviewer_id": self.reviewer_id,
            "review_confidence": self.review_confidence,
            "meta_review_status": self.meta_review_status,
            "meta_review_feedback": self.meta_review_feedback,
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Creates a task object from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            description=data.get("description"),
            source=data.get("source"),
            priority_score=data.get("priority_score"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            dependencies=data.get("dependencies", []),
            review_status=data.get("review_status", "unreviewed"),
            review_feedback=data.get("review_feedback"),
            reviewer_id=data.get("reviewer_id"),
            review_confidence=data.get("review_confidence", 0.0),
            meta_review_status=data.get("meta_review_status", "unreviewed"),
            meta_review_feedback=data.get("meta_review_feedback"),
            validation_status=data.get("validation_status", "unvalidated"),
            validation_reason=data.get("validation_reason"),
            category=data.get("category", "general"),
        )
    
    def is_approved(self) -> bool:
        """Check if the task has passed all reviews."""
        return (
            self.review_status == "approved" and
            self.validation_status in ("valid", "unvalidated")
        )
    
    def is_rejected(self) -> bool:
        """Check if the task has been rejected at any stage."""
        return (
            self.review_status == "rejected" or
            self.validation_status == "invalid"
        )
    
    def is_gated(self) -> bool:
        """Check if the task is gated (valid but blocked on preconditions)."""
        return (
            self.review_status == "gated" or
            self.validation_status == "gated"
        )
