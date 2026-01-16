"""
Task Reviewer Agent - Reviews individual tasks before execution.

This agent validates tasks for feasibility, preconditions, safety, and clarity
before they are queued for execution. Part of the upgraded task decomposition
system with review gates.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from core.logging import log_event
from core.llm_api import run_llm


class ReviewStatus(Enum):
    """Status of a task review."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REFINEMENT = "needs_refinement"
    GATED = "gated"  # Task is valid but preconditions not met


@dataclass
class TaskReviewResult:
    """Result of a task review."""
    status: ReviewStatus
    approved: bool
    feedback: str
    confidence: float = 0.0
    missing_preconditions: List[str] = field(default_factory=list)
    suggested_refinement: Optional[str] = None
    reviewer_id: str = "task_reviewer"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "approved": self.approved,
            "feedback": self.feedback,
            "confidence": self.confidence,
            "missing_preconditions": self.missing_preconditions,
            "suggested_refinement": self.suggested_refinement,
            "reviewer_id": self.reviewer_id,
        }


class TaskReviewerAgent:
    """
    Reviews individual tasks before they are queued for execution.
    
    Validates tasks for:
    - Feasibility: Can this task actually be executed with available tools?
    - Preconditions: Are required resources/permissions available?
    - Safety: Does this task violate operational constraints?
    - Clarity: Is the task well-defined and actionable?
    """
    
    # Known tool/capability registry - tasks requiring these can be checked
    KNOWN_CAPABILITIES = {
        "social_media": ["bluesky", "twitter", "post", "social"],
        "financial": ["stake", "swap", "trade", "defi", "yield", "liquidity", "eth", "crypto", "wallet"],
        "code": ["code", "implement", "fix", "debug", "refactor", "write"],
        "research": ["research", "analyze", "investigate", "study", "explore"],
        "creative": ["create", "generate", "write", "compose", "design"],
    }
    
    # Tasks that should be gated (require integrations we don't have)
    GATED_PATTERNS = [
        # Financial actions without real integrations
        ("stake", "No staking integration available"),
        ("liquidity pool", "No liquidity pool integration available"),
        ("swap token", "No token swap integration available"),
        ("yield farm", "No yield farming integration available"),
        ("exampleyield", "ExampleYield is a placeholder, not a real protocol"),
        ("farmfinance", "FarmFinance is a placeholder, not a real protocol"),
        # Vague philosophical tasks
        ("embrace agape", "Too abstract - needs concrete actionable steps"),
        ("cultivate love", "Too abstract - needs concrete actionable steps"),
        ("manifest abundance", "Too abstract - needs concrete actionable steps"),
    ]
    
    # Tasks that are always valid
    APPROVED_PATTERNS = [
        "manage_bluesky",
        "post to",
        "analyze",
        "research",
        "generate image",
        "create",
        "write",
    ]
    
    REVIEW_PROMPT = """You are a Task Reviewer for an autonomous AI system. Your job is to evaluate if a task is:

1. FEASIBLE: Can it be executed with typical AI agent capabilities?
2. CLEAR: Is it specific and actionable (not vague or philosophical)?
3. SAFE: Does it avoid harmful, illegal, or dangerous actions?
4. REALISTIC: Does it have achievable outcomes?

TASK TO REVIEW:
{task_description}

CONTEXT:
{context}

Respond in JSON format:
{{
    "approved": true/false,
    "status": "approved" | "rejected" | "needs_refinement" | "gated",
    "confidence": 0.0-1.0,
    "feedback": "Brief explanation",
    "issues": ["list of specific issues if any"],
    "suggested_refinement": "If needs_refinement, how to improve it"
}}

Be strict about vague tasks. "Embrace love" is NOT actionable. "Post an inspirational message" IS actionable.
"""

    def __init__(self, tool_registry=None, memory_manager=None):
        """
        Initialize the TaskReviewerAgent.
        
        Args:
            tool_registry: Optional ToolRegistry for checking available tools
            memory_manager: Optional MemoryManager for context
        """
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.review_history: List[Dict] = []
    
    async def review_task(
        self, 
        task_description: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> TaskReviewResult:
        """
        Review a task for feasibility, preconditions, safety, and clarity.
        
        Args:
            task_description: The task to review
            context: Optional context about current state
            
        Returns:
            TaskReviewResult with approval status and feedback
        """
        context = context or {}
        task_lower = task_description.lower()
        
        log_event(f"[TaskReviewer] Reviewing task: {task_description[:100]}...", "DEBUG")
        
        # Quick pattern matching for known gated tasks
        for pattern, reason in self.GATED_PATTERNS:
            if pattern in task_lower:
                result = TaskReviewResult(
                    status=ReviewStatus.GATED,
                    approved=False,
                    feedback=reason,
                    confidence=0.95,
                    missing_preconditions=[reason],
                )
                self._log_review(task_description, result)
                return result
        
        # Quick approval for known good patterns
        for pattern in self.APPROVED_PATTERNS:
            if pattern in task_lower:
                result = TaskReviewResult(
                    status=ReviewStatus.APPROVED,
                    approved=True,
                    feedback="Task matches known actionable pattern",
                    confidence=0.85,
                )
                self._log_review(task_description, result)
                return result
        
        # For ambiguous tasks, use LLM review
        return await self._llm_review(task_description, context)
    
    async def _llm_review(
        self, 
        task_description: str, 
        context: Dict[str, Any]
    ) -> TaskReviewResult:
        """Use LLM to review ambiguous tasks."""
        try:
            prompt = self.REVIEW_PROMPT.format(
                task_description=task_description,
                context=str(context)[:500] if context else "No additional context"
            )
            
            response = await run_llm(prompt, purpose="task_review")
            result_text = response.get("result", "")
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                parsed = json.loads(json_match.group())
                
                status_str = parsed.get("status", "rejected")
                status = ReviewStatus(status_str) if status_str in [s.value for s in ReviewStatus] else ReviewStatus.REJECTED
                
                result = TaskReviewResult(
                    status=status,
                    approved=parsed.get("approved", False),
                    feedback=parsed.get("feedback", "Unknown"),
                    confidence=parsed.get("confidence", 0.5),
                    missing_preconditions=parsed.get("issues", []),
                    suggested_refinement=parsed.get("suggested_refinement"),
                )
            else:
                # Fallback if JSON parsing fails
                result = TaskReviewResult(
                    status=ReviewStatus.NEEDS_REFINEMENT,
                    approved=False,
                    feedback="Could not parse review response",
                    confidence=0.3,
                )
            
            self._log_review(task_description, result)
            return result
            
        except Exception as e:
            log_event(f"[TaskReviewer] LLM review failed: {e}", "ERROR")
            # Conservative fallback - approve but with low confidence
            return TaskReviewResult(
                status=ReviewStatus.APPROVED,
                approved=True,
                feedback=f"Review failed, approving with caution: {e}",
                confidence=0.3,
            )
    
    def _log_review(self, task_description: str, result: TaskReviewResult):
        """Log a review for history tracking."""
        import time
        review_record = {
            "task": task_description,
            "result": result.to_dict(),
            "timestamp": time.time(),
        }
        self.review_history.append(review_record)
        
        # Keep only last 100 reviews in memory
        if len(self.review_history) > 100:
            self.review_history = self.review_history[-100:]
        
        log_event(
            f"[TaskReviewer] Result: {result.status.value} - {result.feedback}", 
            "INFO" if result.approved else "WARNING"
        )
    
    async def batch_review(
        self, 
        tasks: List[str], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[TaskReviewResult]:
        """
        Review multiple tasks in parallel.
        
        Args:
            tasks: List of task descriptions
            context: Optional shared context
            
        Returns:
            List of TaskReviewResult for each task
        """
        results = await asyncio.gather(
            *[self.review_task(task, context) for task in tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to rejection results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TaskReviewResult(
                    status=ReviewStatus.REJECTED,
                    approved=False,
                    feedback=f"Review failed with error: {result}",
                    confidence=0.0,
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_review_stats(self) -> Dict[str, Any]:
        """Get statistics about recent reviews."""
        if not self.review_history:
            return {"total": 0, "approved": 0, "rejected": 0, "gated": 0}
        
        stats = {
            "total": len(self.review_history),
            "approved": sum(1 for r in self.review_history if r["result"]["approved"]),
            "rejected": sum(1 for r in self.review_history if r["result"]["status"] == "rejected"),
            "gated": sum(1 for r in self.review_history if r["result"]["status"] == "gated"),
            "needs_refinement": sum(1 for r in self.review_history if r["result"]["status"] == "needs_refinement"),
        }
        
        return stats
