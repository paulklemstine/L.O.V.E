"""
Meta Reviewer Agent - Reviews complete plans before execution.

This agent provides holistic oversight of generated plans, checking for
coherence, alignment with goals, resource balance, and overall risk.
Works alongside TaskReviewerAgent as part of the review pipeline.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from core.logging import log_event
from core.llm_api import run_llm


class PlanReviewStatus(Enum):
    """Status of a plan review."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REFINEMENT = "needs_refinement"
    PARTIALLY_APPROVED = "partially_approved"


@dataclass
class PlanReviewResult:
    """Result of a complete plan review."""
    status: PlanReviewStatus
    approved: bool
    feedback: str
    confidence: float = 0.0
    coherence_score: float = 0.0
    alignment_score: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    approved_steps: List[int] = field(default_factory=list)
    rejected_steps: List[int] = field(default_factory=list)
    refinement_suggestions: List[str] = field(default_factory=list)
    reviewer_id: str = "meta_reviewer"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "approved": self.approved,
            "feedback": self.feedback,
            "confidence": self.confidence,
            "coherence_score": self.coherence_score,
            "alignment_score": self.alignment_score,
            "risk_level": self.risk_level,
            "approved_steps": self.approved_steps,
            "rejected_steps": self.rejected_steps,
            "refinement_suggestions": self.refinement_suggestions,
            "reviewer_id": self.reviewer_id,
        }


class MetaReviewerAgent:
    """
    Reviews complete plans before execution begins.
    
    Provides holistic oversight checking for:
    - Coherence: Do the tasks form a logical sequence?
    - Alignment: Do they serve the stated goal?
    - Resource Balance: Are we over-committing in one area?
    - Risk Assessment: Overall risk level of the plan
    """
    
    META_REVIEW_PROMPT = """You are a Meta Reviewer for an autonomous AI system. Your job is to evaluate a COMPLETE PLAN holistically.

GOAL:
{goal}

PLAN STEPS:
{plan_steps}

Evaluate the plan on these criteria:

1. COHERENCE (0-1): Do the steps form a logical sequence? Are there gaps or contradictions?
2. ALIGNMENT (0-1): Do all steps actually serve the stated goal?
3. BALANCE: Is the plan focused, or trying to do too many unrelated things?
4. RISK: What's the overall risk level? (low/medium/high/critical)
5. ACTIONABILITY: Can each step actually be executed by an AI agent?

Respond in JSON format:
{{
    "approved": true/false,
    "status": "approved" | "rejected" | "needs_refinement" | "partially_approved",
    "confidence": 0.0-1.0,
    "coherence_score": 0.0-1.0,
    "alignment_score": 0.0-1.0,
    "risk_level": "low" | "medium" | "high" | "critical",
    "feedback": "Brief overall assessment",
    "approved_steps": [list of step indices that are good, 0-indexed],
    "rejected_steps": [list of step indices that are problematic, 0-indexed],
    "refinement_suggestions": ["list of specific improvements"]
}}

Be strict about plans that:
- Mix unrelated goals (e.g., social media + financial speculation in same plan)
- Contain vague or philosophical steps
- Have high risk with unclear benefits
"""

    # Maximum steps we'll review in a single plan
    MAX_PLAN_SIZE = 20
    
    # Risk escalation thresholds
    FINANCIAL_KEYWORDS = ["stake", "swap", "trade", "invest", "liquidity", "defi", "yield"]
    SECURITY_KEYWORDS = ["password", "key", "secret", "credential", "auth", "token"]
    EXTERNAL_KEYWORDS = ["external", "api", "third-party", "service"]

    def __init__(self, memory_manager=None):
        """
        Initialize the MetaReviewerAgent.
        
        Args:
            memory_manager: Optional MemoryManager for historical context
        """
        self.memory_manager = memory_manager
        self.review_history: List[Dict] = []
    
    async def review_plan(
        self, 
        plan: List[str], 
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PlanReviewResult:
        """
        Review a complete plan for coherence, alignment, and risk.
        
        Args:
            plan: List of plan steps
            goal: The stated goal of the plan
            context: Optional execution context
            
        Returns:
            PlanReviewResult with holistic assessment
        """
        context = context or {}
        
        log_event(f"[MetaReviewer] Reviewing plan with {len(plan)} steps for goal: {goal[:100]}...", "DEBUG")
        
        # Handle empty plans
        if not plan:
            return PlanReviewResult(
                status=PlanReviewStatus.REJECTED,
                approved=False,
                feedback="Plan is empty",
                confidence=1.0,
            )
        
        # Handle oversized plans
        if len(plan) > self.MAX_PLAN_SIZE:
            log_event(f"[MetaReviewer] Plan too large ({len(plan)} steps), truncating for review", "WARNING")
            plan = plan[:self.MAX_PLAN_SIZE]
        
        # Quick heuristic checks before LLM
        heuristic_result = self._heuristic_review(plan, goal)
        if heuristic_result:
            self._log_review(plan, goal, heuristic_result)
            return heuristic_result
        
        # Full LLM review
        return await self._llm_review(plan, goal, context)
    
    def _heuristic_review(
        self, 
        plan: List[str], 
        goal: str
    ) -> Optional[PlanReviewResult]:
        """
        Quick heuristic checks for common plan issues.
        Returns a result if issues found, None if LLM review needed.
        """
        plan_text = " ".join(plan).lower()
        
        # Check for high-risk content
        risk_level = "low"
        risk_reasons = []
        
        financial_count = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in plan_text)
        if financial_count >= 2:
            risk_level = "high"
            risk_reasons.append("Multiple financial operations in plan")
        
        security_count = sum(1 for kw in self.SECURITY_KEYWORDS if kw in plan_text)
        if security_count >= 1:
            risk_level = "high" if risk_level == "low" else "critical"
            risk_reasons.append("Security-sensitive operations in plan")
        
        # Check for obviously bad plans
        vague_steps = []
        for i, step in enumerate(plan):
            step_lower = step.lower()
            if any(phrase in step_lower for phrase in [
                "embrace", "cultivate", "manifest",
                "evolve spiritually", "transcend",
            ]):
                vague_steps.append(i)
        
        if len(vague_steps) > len(plan) / 2:
            return PlanReviewResult(
                status=PlanReviewStatus.REJECTED,
                approved=False,
                feedback="Plan contains too many vague/philosophical steps",
                confidence=0.9,
                risk_level=risk_level,
                rejected_steps=vague_steps,
                refinement_suggestions=["Replace abstract concepts with concrete actions"],
            )
        
        # Check for incoherent mixing
        has_social = any("social" in s.lower() or "post" in s.lower() or "bluesky" in s.lower() for s in plan)
        has_financial = any(kw in plan_text for kw in self.FINANCIAL_KEYWORDS)
        has_code = any("code" in s.lower() or "implement" in s.lower() for s in plan)
        
        domain_count = sum([has_social, has_financial, has_code])
        if domain_count >= 3:
            return PlanReviewResult(
                status=PlanReviewStatus.NEEDS_REFINEMENT,
                approved=False,
                feedback="Plan mixes too many unrelated domains - split into focused sub-plans",
                confidence=0.8,
                coherence_score=0.3,
                risk_level=risk_level,
                refinement_suggestions=[
                    "Focus plan on a single domain",
                    "Split into separate social/financial/technical plans",
                ],
            )
        
        return None  # Needs LLM review
    
    async def _llm_review(
        self, 
        plan: List[str], 
        goal: str,
        context: Dict[str, Any]
    ) -> PlanReviewResult:
        """Use LLM for comprehensive plan review."""
        try:
            # Format plan steps
            plan_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            
            prompt = self.META_REVIEW_PROMPT.format(
                goal=goal,
                plan_steps=plan_steps
            )
            
            response = await run_llm(prompt, purpose="meta_review")
            result_text = response.get("result", "")
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                parsed = json.loads(json_match.group())
                
                status_str = parsed.get("status", "rejected")
                status_map = {
                    "approved": PlanReviewStatus.APPROVED,
                    "rejected": PlanReviewStatus.REJECTED,
                    "needs_refinement": PlanReviewStatus.NEEDS_REFINEMENT,
                    "partially_approved": PlanReviewStatus.PARTIALLY_APPROVED,
                }
                status = status_map.get(status_str, PlanReviewStatus.REJECTED)
                
                result = PlanReviewResult(
                    status=status,
                    approved=parsed.get("approved", False),
                    feedback=parsed.get("feedback", "Unknown"),
                    confidence=parsed.get("confidence", 0.5),
                    coherence_score=parsed.get("coherence_score", 0.5),
                    alignment_score=parsed.get("alignment_score", 0.5),
                    risk_level=parsed.get("risk_level", "medium"),
                    approved_steps=parsed.get("approved_steps", []),
                    rejected_steps=parsed.get("rejected_steps", []),
                    refinement_suggestions=parsed.get("refinement_suggestions", []),
                )
            else:
                result = PlanReviewResult(
                    status=PlanReviewStatus.NEEDS_REFINEMENT,
                    approved=False,
                    feedback="Could not parse review response",
                    confidence=0.3,
                )
            
            self._log_review(plan, goal, result)
            return result
            
        except Exception as e:
            log_event(f"[MetaReviewer] LLM review failed: {e}", "ERROR")
            # Conservative fallback
            return PlanReviewResult(
                status=PlanReviewStatus.NEEDS_REFINEMENT,
                approved=False,
                feedback=f"Review failed: {e}",
                confidence=0.2,
                risk_level="medium",
            )
    
    def _log_review(
        self, 
        plan: List[str], 
        goal: str, 
        result: PlanReviewResult
    ):
        """Log a review for history tracking."""
        import time
        review_record = {
            "goal": goal,
            "plan_size": len(plan),
            "result": result.to_dict(),
            "timestamp": time.time(),
        }
        self.review_history.append(review_record)
        
        # Keep only last 50 reviews
        if len(self.review_history) > 50:
            self.review_history = self.review_history[-50:]
        
        log_event(
            f"[MetaReviewer] Result: {result.status.value} (coherence={result.coherence_score:.2f}, risk={result.risk_level})", 
            "INFO" if result.approved else "WARNING"
        )
    
    async def review_with_context(
        self,
        plan: List[str],
        goal: str,
        previous_failures: List[str] = None,
    ) -> PlanReviewResult:
        """
        Review a plan with context from previous failures.
        
        Args:
            plan: The plan to review
            goal: The stated goal
            previous_failures: List of previous failure reasons for this goal
            
        Returns:
            PlanReviewResult with enhanced feedback
        """
        context = {}
        if previous_failures:
            context["previous_failures"] = previous_failures
            context["reflexion_mode"] = True
        
        return await self.review_plan(plan, goal, context)
    
    def get_review_stats(self) -> Dict[str, Any]:
        """Get statistics about recent plan reviews."""
        if not self.review_history:
            return {"total": 0}
        
        stats = {
            "total": len(self.review_history),
            "approved": sum(1 for r in self.review_history if r["result"]["approved"]),
            "avg_coherence": sum(r["result"]["coherence_score"] for r in self.review_history) / len(self.review_history),
            "avg_alignment": sum(r["result"]["alignment_score"] for r in self.review_history) / len(self.review_history),
            "high_risk_count": sum(1 for r in self.review_history if r["result"]["risk_level"] in ["high", "critical"]),
        }
        
        return stats
