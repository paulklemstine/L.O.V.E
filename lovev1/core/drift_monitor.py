"""
Story 1.2: "Ralph Wiggum" Drift Detector

Monitors the agent's thought chain for semantic drift from the original goal.
Triggers "Stop & Reflect" when the agent strays too far from its mission.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.logging import log_event
from core.semantic_similarity import get_similarity_checker


@dataclass
class DriftResult:
    """Result of a drift check."""
    is_drifted: bool
    similarity: float = 1.0
    action: str = "CONTINUE"
    message: str = ""


class DriftMonitor:
    """
    Compares current thought chain against original goal embedding.
    
    Story 1.2: Every N steps, calculates cosine similarity between
    recent reasoning and the original prompt. If below threshold,
    triggers a "Stop & Reflect" interruption.
    """
    
    # Check frequency (every N steps)
    CHECK_INTERVAL = int(os.environ.get("LOVE_DRIFT_CHECK_INTERVAL", "3"))
    
    # Similarity threshold for drift detection (configurable via env)
    DRIFT_THRESHOLD = float(os.environ.get("LOVE_DRIFT_THRESHOLD", "0.6"))
    
    # Path to evolution log for drift events
    EVOLUTION_LOG_PATH = os.environ.get(
        "LOVE_EVOLUTION_LOG",
        "EVOLUTION_LOG.md"
    )
    
    def __init__(self, similarity_threshold: float = None):
        """
        Initialize the DriftMonitor.
        
        Args:
            similarity_threshold: Override for default threshold (0.6)
        """
        self.threshold = similarity_threshold or self.DRIFT_THRESHOLD
        self.similarity_checker = None  # Lazy load
        self.drift_events: List[dict] = []
        self._last_check_step = -1
    
    def _get_similarity_checker(self):
        """Lazy load the semantic similarity checker."""
        if self.similarity_checker is None:
            self.similarity_checker = get_similarity_checker()
        return self.similarity_checker
    
    def check_drift(
        self, 
        original_goal: str, 
        current_thoughts: List[str],
        step_num: int
    ) -> DriftResult:
        """
        Check if the agent has drifted from the original goal.
        
        Story 1.2: Only checks every CHECK_INTERVAL steps to reduce overhead.
        
        Args:
            original_goal: The original user prompt/goal
            current_thoughts: Recent thought chain entries
            step_num: Current step number in execution
            
        Returns:
            DriftResult with drift status and recommended action
        """
        # Only check at intervals
        if step_num % self.CHECK_INTERVAL != 0:
            return DriftResult(is_drifted=False, action="CONTINUE")
        
        # Avoid double-checking same step
        if step_num == self._last_check_step:
            return DriftResult(is_drifted=False, action="CONTINUE")
        
        self._last_check_step = step_num
        
        if not current_thoughts:
            return DriftResult(is_drifted=False, action="CONTINUE")
        
        # Combine recent thoughts for comparison
        # Use last 5 thoughts for recent context
        recent_context = " ".join(current_thoughts[-5:])
        
        if not recent_context.strip():
            return DriftResult(is_drifted=False, action="CONTINUE")
        
        try:
            checker = self._get_similarity_checker()
            result = checker.check_similarity(original_goal, recent_context)
            similarity = result.get("similarity", 1.0)
            
            if similarity < self.threshold:
                drift_event = {
                    "step": step_num,
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "original_goal": original_goal[:100],
                    "recent_context": recent_context[:200],
                    "timestamp": datetime.now().isoformat()
                }
                
                self._log_drift_event(drift_event)
                
                return DriftResult(
                    is_drifted=True,
                    similarity=similarity,
                    action="STOP_AND_REFLECT",
                    message=(
                        f"Drift detected at step {step_num}: "
                        f"similarity {similarity:.2f} < {self.threshold:.2f}. "
                        f"Agent should pause and re-evaluate approach."
                    )
                )
            
            log_event(
                f"DriftMonitor: Step {step_num} similarity {similarity:.2f} (OK)",
                "DEBUG"
            )
            
            return DriftResult(
                is_drifted=False,
                similarity=similarity,
                action="CONTINUE"
            )
            
        except Exception as e:
            log_event(f"DriftMonitor: Check failed - {e}", "WARNING")
            # On error, don't block execution
            return DriftResult(is_drifted=False, action="CONTINUE")
    
    def _log_drift_event(self, event: dict) -> None:
        """
        Log drift event to EVOLUTION_LOG.md for traceability.
        
        Story 1.2: All drift events must be logged.
        """
        self.drift_events.append(event)
        
        log_event(
            f"DriftMonitor: DRIFT DETECTED at step {event['step']} "
            f"(similarity: {event['similarity']:.2f})",
            "WARNING"
        )
        
        # Append to evolution log
        try:
            log_entry = (
                f"| {event['timestamp']} | DRIFT_DETECTED | Step {event['step']} | "
                f"Similarity: {event['similarity']:.2f} | {event['recent_context'][:50]}... |\n"
            )
            
            with open(self.EVOLUTION_LOG_PATH, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            log_event(f"Failed to write to evolution log: {e}", "WARNING")
    
    def get_drift_summary(self) -> str:
        """Returns summary of all drift events in this session."""
        if not self.drift_events:
            return "No drift events detected"
        
        summary = f"Drift Events ({len(self.drift_events)} total):\n"
        for event in self.drift_events[-5:]:  # Last 5
            summary += (
                f"  - Step {event['step']}: "
                f"similarity {event['similarity']:.2f}\n"
            )
        
        return summary
    
    def reset(self) -> None:
        """Reset drift monitor state for new task."""
        self.drift_events = []
        self._last_check_step = -1
        log_event("DriftMonitor: Reset for new task", "DEBUG")


# Global instance for easy access
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create the global DriftMonitor instance."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


def check_thought_drift(
    original_goal: str,
    thoughts: List[str],
    step: int
) -> DriftResult:
    """
    Convenience function to check for drift.
    
    Usage in execution loop:
        drift = check_thought_drift(user_prompt, thought_chain, step_num)
        if drift.is_drifted:
            # Trigger reflection
    """
    return get_drift_monitor().check_drift(original_goal, thoughts, step)
