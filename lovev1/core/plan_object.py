"""
Story 1.1: Plan-Then-Execute Enforcement

Provides an immutable PlanObject structure with hash verification to ensure
execution cannot proceed without a validated plan.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

from core.logging import log_event


@dataclass(frozen=True)
class PlanObject:
    """
    Immutable, hashable plan structure for execution gating.
    
    Story 1.1: GodAgentReActEngine must validate this object before
    executing any steps. The frozen=True ensures immutability.
    """
    plan_id: str
    steps: Tuple[str, ...]  # Immutable tuple
    intent_hash: str        # SHA256 of original user prompt
    created_at: float
    context_tokens: int
    original_prompt: str    # For drift detection
    
    @property
    def plan_hash(self) -> str:
        """Unique hash for verification."""
        content = f"{self.plan_id}:{':'.join(self.steps)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    def step_count(self) -> int:
        return len(self.steps)
    
    def get_step(self, index: int) -> Optional[str]:
        """Safely get a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "steps": list(self.steps),
            "intent_hash": self.intent_hash,
            "plan_hash": self.plan_hash,
            "created_at": self.created_at,
            "context_tokens": self.context_tokens,
            "original_prompt": self.original_prompt[:200]  # Truncate for storage
        }


class PlanValidationError(Exception):
    """Raised when plan validation fails."""
    pass


class ReplanRequired(Exception):
    """Raised when context window threshold exceeded."""
    pass


class PlanManager:
    """
    Manages plan lifecycle: creation, validation, revision, and context monitoring.
    
    Story 1.1: Enforces plan-then-execute pattern by gating execution.
    """
    
    # Context window threshold for forced replan (70%)
    CONTEXT_THRESHOLD = float(os.environ.get("LOVE_CONTEXT_THRESHOLD", "0.7"))
    
    def __init__(self, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.active_plan: Optional[PlanObject] = None
        self.plan_history: List[Dict[str, Any]] = []
        self._validated_hashes: set = set()
    
    def create_plan(
        self, 
        user_prompt: str, 
        steps: List[str],
        context_tokens: int = 0
    ) -> PlanObject:
        """
        Creates a new immutable PlanObject.
        
        Args:
            user_prompt: Original user request
            steps: List of atomic steps from PlannerAgent
            context_tokens: Current context window usage
            
        Returns:
            Validated PlanObject
        """
        plan = PlanObject(
            plan_id=str(uuid.uuid4()),
            steps=tuple(steps),  # Convert to immutable tuple
            intent_hash=hashlib.sha256(user_prompt.encode()).hexdigest(),
            created_at=datetime.now().timestamp(),
            context_tokens=context_tokens,
            original_prompt=user_prompt
        )
        
        # Auto-validate new plans
        self._validated_hashes.add(plan.plan_hash)
        self.active_plan = plan
        
        log_event(
            f"PlanManager: Created plan {plan.plan_id[:8]} with {len(steps)} steps",
            "INFO"
        )
        
        return plan
    
    def validate_plan(self, plan: PlanObject) -> bool:
        """
        Verifies plan signature and that it hasn't been tampered with.
        
        Story 1.1: Execution blocked if validation fails.
        
        Args:
            plan: PlanObject to validate
            
        Returns:
            True if valid
        """
        if plan is None:
            log_event("PlanManager: Validation failed - no plan provided", "WARNING")
            return False
        
        if plan.plan_hash not in self._validated_hashes:
            log_event(
                f"PlanManager: Validation failed - unknown plan hash {plan.plan_hash[:16]}",
                "WARNING"
            )
            return False
        
        # Verify integrity by recalculating hash
        expected_hash = hashlib.sha256(
            f"{plan.plan_id}:{':'.join(plan.steps)}".encode()
        ).hexdigest()
        
        if expected_hash != plan.plan_hash:
            log_event("PlanManager: Validation failed - hash mismatch (tampering?)", "ERROR")
            return False
        
        return True
    
    def should_replan(self, current_context_tokens: int) -> bool:
        """
        Checks if context window usage exceeds threshold.
        
        Story 1.1: Forces replan at 70% context to prevent drift.
        
        Args:
            current_context_tokens: Current token count
            
        Returns:
            True if replan needed
        """
        threshold = int(self.max_context_tokens * self.CONTEXT_THRESHOLD)
        
        if current_context_tokens >= threshold:
            log_event(
                f"PlanManager: Context at {current_context_tokens}/{self.max_context_tokens} "
                f"({current_context_tokens/self.max_context_tokens:.1%}). Replan required.",
                "WARNING"
            )
            return True
        
        return False
    
    def request_revision(
        self, 
        current_plan: PlanObject, 
        reason: str,
        new_steps: List[str],
        new_context_tokens: int = 0
    ) -> PlanObject:
        """
        Creates a new plan as explicit revision with lineage tracking.
        
        Story 1.1: Only way to modify a plan - creates new immutable object.
        
        Args:
            current_plan: Plan being revised
            reason: Why revision was needed
            new_steps: Updated step list
            new_context_tokens: Updated context count
            
        Returns:
            New PlanObject with lineage
        """
        # Archive current plan
        if current_plan:
            archive_entry = current_plan.to_dict()
            archive_entry["revision_reason"] = reason
            archive_entry["revised_at"] = datetime.now().isoformat()
            self.plan_history.append(archive_entry)
        
        log_event(
            f"PlanManager: Revision requested - {reason}. Creating new plan.",
            "INFO"
        )
        
        # Create new plan with same original prompt (for drift detection)
        return self.create_plan(
            user_prompt=current_plan.original_prompt if current_plan else "",
            steps=new_steps,
            context_tokens=new_context_tokens
        )
    
    def invalidate_plan(self, plan: PlanObject) -> None:
        """Removes a plan from validation registry (e.g., on task completion)."""
        if plan and plan.plan_hash in self._validated_hashes:
            self._validated_hashes.discard(plan.plan_hash)
            log_event(f"PlanManager: Invalidated plan {plan.plan_id[:8]}", "DEBUG")
    
    def get_plan_summary(self) -> str:
        """Returns human-readable summary of active plan."""
        if not self.active_plan:
            return "No active plan"
        
        steps_preview = "\n".join(
            f"  {i+1}. {step[:60]}..." if len(step) > 60 else f"  {i+1}. {step}"
            for i, step in enumerate(self.active_plan.steps)
        )
        
        return (
            f"Plan {self.active_plan.plan_id[:8]} ({self.active_plan.step_count} steps):\n"
            f"{steps_preview}"
        )


# Convenience functions
def create_plan_from_prompt(prompt: str, steps: List[str]) -> PlanObject:
    """Quick plan creation without PlanManager instance."""
    manager = PlanManager()
    return manager.create_plan(prompt, steps)
