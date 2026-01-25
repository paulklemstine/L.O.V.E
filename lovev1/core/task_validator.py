"""
Task Validator - Validates tasks before they can be queued.

This module provides precondition checking and validation for tasks,
preventing impractical or dangerous tasks from being executed.
Rejected tasks are stored in rejected_goals for future reference.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from enum import Enum

from core.logging import log_event


class ValidationStatus(Enum):
    """Status of task validation."""
    VALID = "valid"
    INVALID = "invalid"
    GATED = "gated"  # Valid but preconditions not met
    NEEDS_INFO = "needs_info"  # Requires more information


@dataclass
class ValidationResult:
    """Result of a task validation."""
    valid: bool
    status: ValidationStatus
    reason: str
    category: str = "general"
    missing_preconditions: List[str] = field(default_factory=list)
    should_log_as_rejected: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "reason": self.reason,
            "category": self.category,
            "missing_preconditions": self.missing_preconditions,
            "should_log_as_rejected": self.should_log_as_rejected,
            "metadata": self.metadata,
        }


class FinancialTaskValidator:
    """Validates financial/crypto-related tasks."""
    
    # Known placeholder protocols that aren't real
    PLACEHOLDER_PROTOCOLS = [
        "exampleyield", "farmfinance", "defiexample", 
        "cryptotest", "mockstake", "testpool"
    ]
    
    # Real integrations we have (add to this as integrations are built)
    REAL_INTEGRATIONS = [
        # Currently empty - no real DeFi integrations
    ]
    
    async def validate(self, task: str, context: Dict) -> ValidationResult:
        """Validate a financial task."""
        task_lower = task.lower()
        
        # Check for placeholder protocols
        for placeholder in self.PLACEHOLDER_PROTOCOLS:
            if placeholder in task_lower:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.GATED,
                    reason=f"'{placeholder}' is a placeholder, not a real protocol",
                    category="financial",
                    missing_preconditions=[f"Real integration with {placeholder}"],
                )
        
        # Check if we have required wallet/funds
        if any(word in task_lower for word in ["stake", "swap", "trade", "liquidity"]):
            # Check for wallet configuration
            wallet_configured = context.get("wallet_configured", False)
            has_funds = context.get("has_crypto_funds", False)
            
            if not wallet_configured:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.GATED,
                    reason="No wallet configured for financial operations",
                    category="financial",
                    missing_preconditions=["Wallet configuration", "Private key setup"],
                )
            
            if not has_funds:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.GATED,
                    reason="No funds available for financial operations",
                    category="financial",
                    missing_preconditions=["Funds in wallet"],
                )
        
        # Check for real integrations
        has_real_integration = any(
            integration in task_lower 
            for integration in self.REAL_INTEGRATIONS
        )
        
        if not has_real_integration and any(
            word in task_lower 
            for word in ["defi", "yield", "farm", "liquidity pool"]
        ):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.GATED,
                reason="No DeFi protocol integrations available",
                category="financial",
                missing_preconditions=["DeFi protocol integration"],
            )
        
        return ValidationResult(
            valid=True,
            status=ValidationStatus.VALID,
            reason="Financial task preliminarily valid",
            category="financial",
        )


class SocialTaskValidator:
    """Validates social media tasks."""
    
    async def validate(self, task: str, context: Dict) -> ValidationResult:
        """Validate a social media task."""
        task_lower = task.lower()
        
        # Check for social media credentials
        if "bluesky" in task_lower:
            if not context.get("bluesky_configured", True):  # Assume configured by default
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.GATED,
                    reason="Bluesky credentials not configured",
                    category="social",
                    missing_preconditions=["BLUESKY_HANDLE", "BLUESKY_PASSWORD"],
                )
        
        if "twitter" in task_lower or "x.com" in task_lower:
            if not context.get("twitter_configured", False):
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.GATED,
                    reason="Twitter/X integration not available",
                    category="social",
                    missing_preconditions=["Twitter API keys"],
                )
        
        return ValidationResult(
            valid=True,
            status=ValidationStatus.VALID,
            reason="Social media task valid",
            category="social",
        )


class TechnicalTaskValidator:
    """Validates code/technical tasks."""
    
    async def validate(self, task: str, context: Dict) -> ValidationResult:
        """Validate a technical task."""
        # Technical tasks are generally valid if they're well-formed
        task_lower = task.lower()
        
        # Check for dangerous operations
        dangerous_patterns = [
            ("rm -rf", "Destructive file deletion"),
            ("format", "Disk formatting"),
            ("delete database", "Database deletion"),
            ("drop table", "Database table deletion"),
        ]
        
        for pattern, reason in dangerous_patterns:
            if pattern in task_lower:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.INVALID,
                    reason=f"Dangerous operation detected: {reason}",
                    category="technical",
                    should_log_as_rejected=True,
                )
        
        return ValidationResult(
            valid=True,
            status=ValidationStatus.VALID,
            reason="Technical task valid",
            category="technical",
        )


class AbstractTaskValidator:
    """Validates abstract/philosophical tasks (usually rejects them)."""
    
    ABSTRACT_PATTERNS = [
        ("embrace agape", "Too abstract - 'embrace' is not actionable"),
        ("cultivate love", "Too abstract - needs concrete steps"),
        ("manifest abundance", "Too abstract - needs specific actions"),
        ("evolve spiritually", "Too abstract - needs measurable outcomes"),
        ("transcend", "Too abstract - not actionable"),
        ("become one with", "Too abstract - not actionable"),
        ("tap into", "Too vague - needs specific mechanism"),
        ("harness the power", "Too vague - needs specific steps"),
    ]
    
    async def validate(self, task: str, context: Dict) -> ValidationResult:
        """Validate an abstract task."""
        task_lower = task.lower()
        
        for pattern, reason in self.ABSTRACT_PATTERNS:
            if pattern in task_lower:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.INVALID,
                    reason=reason,
                    category="abstract",
                    should_log_as_rejected=True,
                )
        
        return ValidationResult(
            valid=True,
            status=ValidationStatus.VALID,
            reason="Task is concrete enough",
            category="general",
        )


class TaskValidator:
    """
    Main validator that orchestrates validation across categories.
    """
    
    # Path for storing rejected goals
    REJECTED_GOALS_PATH = Path(".agent_workspace") / "rejected_goals.json"
    
    def __init__(self, context_provider: Callable[[], Dict] = None):
        """
        Initialize the TaskValidator.
        
        Args:
            context_provider: Optional callable that returns current context
        """
        self.context_provider = context_provider
        self.validators = {
            "financial": FinancialTaskValidator(),
            "social": SocialTaskValidator(),
            "technical": TechnicalTaskValidator(),
            "abstract": AbstractTaskValidator(),
        }
        self._ensure_workspace()
    
    def _ensure_workspace(self):
        """Ensure the agent workspace directory exists."""
        self.REJECTED_GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    def _classify_task(self, task: str) -> str:
        """Classify a task into a category."""
        task_lower = task.lower()
        
        # Financial keywords
        if any(word in task_lower for word in [
            "stake", "swap", "trade", "liquidity", "defi", 
            "yield", "crypto", "eth", "token", "wallet"
        ]):
            return "financial"
        
        # Social keywords
        if any(word in task_lower for word in [
            "post", "tweet", "bluesky", "social", "media",
            "share", "engage", "comment", "reply"
        ]):
            return "social"
        
        # Technical keywords
        if any(word in task_lower for word in [
            "code", "implement", "fix", "debug", "refactor",
            "deploy", "test", "build", "compile"
        ]):
            return "technical"
        
        # Abstract keywords (check these to reject)
        if any(word in task_lower for word in [
            "embrace", "cultivate", "manifest", "evolve",
            "transcend", "spiritual", "consciousness"
        ]):
            return "abstract"
        
        return "general"
    
    async def validate(
        self, 
        task_description: str,
        context: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate a task before queuing.
        
        Args:
            task_description: The task to validate
            context: Optional context dict
            
        Returns:
            ValidationResult with validation status
        """
        # Safety check: Reject system metadata that shouldn't be tasks
        # These prefixes indicate internal system messages, not actionable tasks
        SYSTEM_PREFIXES = [
            "Reflexion Adjustment:", 
            "Strategic Insight:", 
            "Meta Review:",
            "Insight:",
            "CRITICAL BLOCKER:",
        ]
        
        for prefix in SYSTEM_PREFIXES:
            if task_description.startswith(prefix):
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.INVALID,
                    reason=f"System metadata ('{prefix}'), not a task",
                    category="system",
                    should_log_as_rejected=False,  # Don't pollute rejected_goals log
                )
        
        # Get context
        if context is None:
            context = self.context_provider() if self.context_provider else {}
        
        # Classify the task
        category = self._classify_task(task_description)
        
        log_event(f"[TaskValidator] Validating '{category}' task: {task_description[:80]}...", "DEBUG")
        
        # First, check abstract validator (applies to all)
        abstract_result = await self.validators["abstract"].validate(task_description, context)
        if not abstract_result.valid:
            await self._log_rejected_goal(task_description, abstract_result)
            return abstract_result
        
        # Then check category-specific validator
        if category in self.validators:
            result = await self.validators[category].validate(task_description, context)
            if not result.valid:
                await self._log_rejected_goal(task_description, result)
            return result
        
        # Default valid for uncategorized tasks
        return ValidationResult(
            valid=True,
            status=ValidationStatus.VALID,
            reason="Task passed validation",
            category=category,
        )
    
    async def _log_rejected_goal(
        self, 
        task: str, 
        result: ValidationResult
    ):
        """Log a rejected goal for future reference."""
        if not result.should_log_as_rejected:
            return
        
        try:
            rejected_goals = []
            if self.REJECTED_GOALS_PATH.exists():
                with open(self.REJECTED_GOALS_PATH, "r") as f:
                    rejected_goals = json.load(f)
            
            rejected_goals.append({
                "task": task,
                "reason": result.reason,
                "category": result.category,
                "status": result.status.value,
                "missing_preconditions": result.missing_preconditions,
                "timestamp": time.time(),
            })
            
            # Keep only last 500 rejected goals
            if len(rejected_goals) > 500:
                rejected_goals = rejected_goals[-500:]
            
            with open(self.REJECTED_GOALS_PATH, "w") as f:
                json.dump(rejected_goals, f, indent=2)
            
            log_event(f"[TaskValidator] Logged rejected goal: {task[:50]}...", "INFO")
            
        except Exception as e:
            log_event(f"[TaskValidator] Failed to log rejected goal: {e}", "ERROR")
    
    async def batch_validate(
        self, 
        tasks: List[str],
        context: Optional[Dict] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple tasks.
        
        Args:
            tasks: List of task descriptions
            context: Optional shared context
            
        Returns:
            List of ValidationResults
        """
        results = await asyncio.gather(
            *[self.validate(task, context) for task in tasks]
        )
        return list(results)
    
    def get_rejected_goals(self, limit: int = 50) -> List[Dict]:
        """Get recent rejected goals."""
        try:
            if self.REJECTED_GOALS_PATH.exists():
                with open(self.REJECTED_GOALS_PATH, "r") as f:
                    goals = json.load(f)
                    return goals[-limit:]
        except Exception as e:
            log_event(f"[TaskValidator] Failed to read rejected goals: {e}", "ERROR")
        return []
    
    def get_rejection_stats(self) -> Dict[str, Any]:
        """Get statistics about rejected goals."""
        goals = self.get_rejected_goals(limit=500)
        
        if not goals:
            return {"total": 0}
        
        categories = {}
        reasons = {}
        
        for goal in goals:
            cat = goal.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            reason = goal.get("status", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            "total": len(goals),
            "by_category": categories,
            "by_status": reasons,
        }
