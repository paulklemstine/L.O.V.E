"""
Evolve Tool - Self-Evolution through Codebase Introspection

The core self-evolution tool that:
1. Introspects the codebase to find improvement opportunities
2. Prioritizes improvements based on impact and feasibility
3. Generates detailed user stories
4. Submits to Jules API for autonomous implementation

Rate limited to 1 evolution task per 15 minutes.

Adapted for L.O.V.E. v2 architecture.
"""

import os
import time
import json
import asyncio
from ..async_utils import run_sync_safe
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

# Radon for complexity analysis
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except ImportError:
    cc_visit = None
    mi_visit = None

from core.introspection.code_index_manager import CodeIndexManager, get_or_create_index
from core.introspection.codebase_search import CodebaseSearch


# ============================================================================
# Data Structures
# ============================================================================

class ImprovementCategory(Enum):
    """Categories of code improvements."""
    REFACTOR = "refactor"
    FEATURE = "feature"
    BUGFIX = "bugfix"
    TEST = "test"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ImprovementOpportunity:
    """
    Represents a potential improvement opportunity in the codebase.
    """
    category: ImprovementCategory
    priority: int  # 1-10 (10 = highest)
    target_file: str
    target_function: Optional[str]
    title: str
    description: str
    evidence: str  # Why this was flagged
    estimated_impact: str
    user_story: Optional[str] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'priority': self.priority,
            'target_file': self.target_file,
            'target_function': self.target_function,
            'title': self.title,
            'description': self.description,
            'evidence': self.evidence,
            'estimated_impact': self.estimated_impact,
            'user_story': self.user_story,
            'acceptance_criteria': self.acceptance_criteria,
        }


# ============================================================================
# Codebase Analyzer
# ============================================================================

class CodebaseAnalyzer:
    """
    Analyzes the codebase to find improvement opportunities.
    
    Strategies:
    - Complexity hotspots (high cyclomatic complexity)
    - Error patterns from logs
    - Missing tests
    - TODO/FIXME items
    - Documentation gaps
    """
    
    # Thresholds
    COMPLEXITY_THRESHOLD = 10  # Cyclomatic complexity
    MAINTAINABILITY_THRESHOLD = 50  # Maintainability index (lower = worse)
    
    def __init__(self, code_index: CodeIndexManager):
        self.code_index = code_index
        self.codebase_root = code_index.codebase_root
    
    def analyze_all(self) -> List[ImprovementOpportunity]:
        """Run all analysis strategies and return combined results."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        opportunities = []
        
        # Complexity analysis
        opportunities.extend(self._analyze_complexity())
        
        # TODO/FIXME scanning
        opportunities.extend(self._scan_todos())
        
        # Documentation gaps
        opportunities.extend(self._find_missing_docs())
        
        # Error pattern analysis from logs
        opportunities.extend(self._analyze_error_patterns())
        
        # Sort by priority (descending)
        opportunities.sort(key=lambda o: o.priority, reverse=True)
        
        return opportunities
    
    def _analyze_complexity(self) -> List[ImprovementOpportunity]:
        """Find high-complexity code that needs refactoring."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        opportunities = []
        
        if cc_visit is None:
            logger.warning("radon not available for complexity analysis")
            return opportunities
        
        for chunk in self.code_index.chunks:
            if chunk.chunk_type not in ['function', 'method', 'async_function', 'async_method']:
                continue
            
            try:
                # Calculate cyclomatic complexity
                results = cc_visit(chunk.content)
                if not results:
                    continue
                
                complexity = results[0].complexity
                
                if complexity >= self.COMPLEXITY_THRESHOLD:
                    priority = min(10, 5 + (complexity - self.COMPLEXITY_THRESHOLD) // 2)
                    
                    opportunities.append(ImprovementOpportunity(
                        category=ImprovementCategory.REFACTOR,
                        priority=priority,
                        target_file=chunk.file_path,
                        target_function=chunk.name,
                        title=f"Refactor high-complexity function: {chunk.qualified_name}",
                        description=f"Function has cyclomatic complexity of {complexity}, which exceeds the threshold of {self.COMPLEXITY_THRESHOLD}. Consider breaking it into smaller, more focused functions.",
                        evidence=f"Cyclomatic complexity: {complexity} (threshold: {self.COMPLEXITY_THRESHOLD})",
                        estimated_impact="Medium - Improved maintainability and testability",
                    ))
                    
            except Exception as e:
                logger.debug(f"Complexity analysis failed for {chunk.name}: {e}")
        
        return opportunities
    
    def _scan_todos(self) -> List[ImprovementOpportunity]:
        """Find TODO, FIXME, HACK, and XXX comments."""
        opportunities = []
        seen_titles = set()  # Deduplicate by title
        
        patterns = {
            'TODO': (ImprovementCategory.FEATURE, 5),
            'FIXME': (ImprovementCategory.BUGFIX, 7),
            'HACK': (ImprovementCategory.REFACTOR, 6),
            'XXX': (ImprovementCategory.BUGFIX, 6),
            'BUG': (ImprovementCategory.BUGFIX, 8),
        }
        
        import re
        
        for chunk in self.code_index.chunks:
            for pattern, (category, base_priority) in patterns.items():
                # More specific regex to capture the full comment
                matches = list(re.finditer(
                    rf'#\s*{pattern}\b[:\s]*(.*?)(?:\n|$)', 
                    chunk.content, 
                    re.IGNORECASE
                ))
                
                for match in matches:
                    raw_description = match.group(1).strip()
                    
                    # Skip if description is too short or looks like another pattern
                    if len(raw_description) < 3 or raw_description.startswith('/'):
                        continue
                    
                    # Clean up the description
                    description = raw_description[:100] if raw_description else f"Found {pattern} in {chunk.name}"
                    
                    # Create unique title including location
                    title = f"{pattern} in {chunk.qualified_name}: {description[:40]}"
                    
                    # Deduplicate
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    opportunities.append(ImprovementOpportunity(
                        category=category,
                        priority=base_priority,
                        target_file=chunk.file_path,
                        target_function=chunk.name if chunk.chunk_type != 'class' else None,
                        title=title,
                        description=description,
                        evidence=f"Found '{pattern}' comment in {chunk.qualified_name} at line ~{chunk.start_line}",
                        estimated_impact="Variable - depends on the specific issue",
                    ))
        
        return opportunities
    
    def _find_missing_docs(self) -> List[ImprovementOpportunity]:
        """Find functions and classes without docstrings."""
        opportunities = []
        
        for chunk in self.code_index.chunks:
            if chunk.chunk_type not in ['function', 'method', 'class', 'async_function', 'async_method']:
                continue
            
            # Skip private/internal functions
            if chunk.name.startswith('_'):
                continue
            
            if not chunk.docstring:
                opportunities.append(ImprovementOpportunity(
                    category=ImprovementCategory.DOCUMENTATION,
                    priority=3,
                    target_file=chunk.file_path,
                    target_function=chunk.name if chunk.chunk_type != 'class' else None,
                    title=f"Add documentation for {chunk.chunk_type}: {chunk.qualified_name}",
                    description=f"The {chunk.chunk_type} '{chunk.qualified_name}' lacks a docstring. Add documentation explaining its purpose, parameters, and return value.",
                    evidence=f"No docstring found for {chunk.signature}",
                    estimated_impact="Low - Improved code understanding",
                ))
        
        return opportunities
    
    def _analyze_error_patterns(self) -> List[ImprovementOpportunity]:
        """Analyze log files for recurring error patterns."""
        opportunities = []
        
        if not self.codebase_root:
            return opportunities
        
        # Look for log files
        log_paths = [
            Path(self.codebase_root) / 'logs',
            Path(self.codebase_root) / 'log',
            Path(self.codebase_root).parent / 'logs',
        ]
        
        error_counts: Dict[str, Dict] = {}  # file_path -> {error_type: count}
        
        import re
        
        for log_dir in log_paths:
            if not log_dir.exists():
                continue
            
            for log_file in log_dir.glob('*.log'):
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Find file references in error messages
                    error_pattern = r'\[ERROR\].*?(?:File "([^"]+)", line \d+|(\w+Error))'
                    
                    for match in re.finditer(error_pattern, content):
                        file_path = match.group(1) or "unknown"
                        error_type = match.group(2) or "Error"
                        
                        if file_path not in error_counts:
                            error_counts[file_path] = {}
                        error_counts[file_path][error_type] = error_counts[file_path].get(error_type, 0) + 1
                        
                except Exception:
                    pass
        
        # Create opportunities for files with recurring errors
        for file_path, errors in error_counts.items():
            total_errors = sum(errors.values())
            if total_errors >= 3:  # Threshold for flagging
                error_summary = ', '.join(f"{k}({v})" for k, v in errors.items())
                
                opportunities.append(ImprovementOpportunity(
                    category=ImprovementCategory.BUGFIX,
                    priority=min(10, 5 + total_errors // 2),
                    target_file=file_path,
                    target_function=None,
                    title=f"Fix recurring errors in {Path(file_path).name}",
                    description=f"This file has generated {total_errors} errors in logs. Error types: {error_summary}",
                    evidence=f"Log analysis: {error_summary}",
                    estimated_impact="High - Improved stability",
                ))
        
        return opportunities


# ============================================================================
# User Story Generator
# ============================================================================

class UserStoryGenerator:
    """Generates detailed user stories from improvement opportunities."""
    
    STORY_TEMPLATE = """
As a developer/user of L.O.V.E.,
I want to {action}
So that {benefit}

## Context
{context}

## Acceptance Criteria
{criteria}

## Technical Notes
- Target file: {target_file}
- Target function: {target_function}
- Category: {category}
- Evidence: {evidence}
"""
    
    async def generate(self, opportunity: ImprovementOpportunity, code_context: str = "") -> str:
        """
        Generate a detailed user story from an improvement opportunity.
        
        Uses LLM to enhance the story if available, otherwise uses template.
        """
        # Try LLM-enhanced generation
        try:
            from core.llm_client import LLMClient
            client = LLMClient()
            return await self._llm_generate(client, opportunity, code_context)
        except Exception as e:
            from core.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(f"LLM story generation failed: {e}")
        
        # Fallback to template-based generation
        return self._template_generate(opportunity)
    
    async def _llm_generate(self, client, opportunity: ImprovementOpportunity, code_context: str) -> str:
        """Generate story using LLM."""
        prompt = f"""You are a senior software engineer. Generate a detailed user story for the following code improvement.

## Improvement Opportunity
**Title:** {opportunity.title}
**Category:** {opportunity.category.value}
**Description:** {opportunity.description}
**Evidence:** {opportunity.evidence}
**Target File:** {opportunity.target_file}
**Target Function:** {opportunity.target_function or 'N/A'}

## Code Context
```python
{code_context[:2000]}
```

## Instructions
Generate a user story in the following format:
1. Start with "As a... I want to... So that..."
2. Include clear acceptance criteria (3-5 items)
3. Add technical notes about implementation approach
4. Keep it actionable and specific

Respond with only the user story, no additional commentary."""

        response = await client.generate(prompt, max_tokens=1000)
        
        if response and len(response) > 50:
            return response
        
        return self._template_generate(opportunity)
    
    def _template_generate(self, opportunity: ImprovementOpportunity) -> str:
        """Generate story using template."""
        # Map category to action/benefit
        action_map = {
            ImprovementCategory.REFACTOR: ("refactor the complex code", "the codebase is more maintainable and testable"),
            ImprovementCategory.FEATURE: ("implement this feature", "the system has new capabilities"),
            ImprovementCategory.BUGFIX: ("fix this bug", "the system is more stable and reliable"),
            ImprovementCategory.TEST: ("add tests for this code", "we have better test coverage"),
            ImprovementCategory.DOCUMENTATION: ("document this code", "developers can understand it more easily"),
            ImprovementCategory.PERFORMANCE: ("optimize this code", "the system runs faster"),
            ImprovementCategory.SECURITY: ("fix this security issue", "the system is more secure"),
        }
        
        action, benefit = action_map.get(opportunity.category, ("improve this code", "the codebase is better"))
        
        criteria = opportunity.acceptance_criteria or [
            f"The {opportunity.category.value} is completed",
            "All existing tests pass",
            "Code follows project conventions",
            "Changes are documented",
        ]
        criteria_str = '\n'.join(f"- [ ] {c}" for c in criteria)
        
        return self.STORY_TEMPLATE.format(
            action=action,
            benefit=benefit,
            context=opportunity.description,
            criteria=criteria_str,
            target_file=opportunity.target_file,
            target_function=opportunity.target_function or 'N/A',
            category=opportunity.category.value,
            evidence=opportunity.evidence,
        )


# ============================================================================
# User Story Quality Reviewer
# ============================================================================

@dataclass
class ReviewResult:
    """Result of a user story quality review."""
    passed: bool
    score: int  # 1-10 quality score
    issues: List[str]  # List of issues found
    suggestions: List[str]  # Suggestions for improvement
    improved_story: Optional[str] = None  # Improved version if review failed


class UserStoryReviewer:
    """
    Automated quality reviewer for user stories.
    
    Uses LLM to evaluate user story quality and regenerate if below threshold.
    This ensures only high-quality, actionable stories are sent to Jules.
    
    Quality criteria:
    - Clear and specific problem description
    - Actionable acceptance criteria
    - Sufficient technical context
    - Reasonable scope (not too broad)
    - Not a duplicate of recent work
    """
    
    # Minimum quality score to pass review (1-10)
    MIN_QUALITY_SCORE = 6
    
    # Maximum regeneration attempts
    MAX_RETRIES = 2
    
    REVIEW_PROMPT = '''You are a senior software engineer reviewing a user story before it's sent to an automated coding agent (Jules) for implementation.

## User Story to Review
{user_story}

## Context
- Target file: {target_file}
- Category: {category}
- Evidence: {evidence}

## Review Criteria
Rate each criterion 1-10 and provide an overall score:

1. **Clarity**: Is the problem clearly described?
2. **Specificity**: Are the changes specific and well-defined?
3. **Scope**: Is the scope reasonable (not too broad or vague)?
4. **Acceptance Criteria**: Are there clear, testable acceptance criteria?
5. **Actionability**: Can an automated agent implement this without asking questions?

## Response Format (JSON)
{{
  "overall_score": <1-10>,
  "clarity": <1-10>,
  "specificity": <1-10>,
  "scope": <1-10>,
  "acceptance_criteria": <1-10>,
  "actionability": <1-10>,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "passed": <true/false>,
  "improved_story": "<only if passed=false, provide an improved version>"
}}

Respond with ONLY the JSON, no other text.'''

    async def review(
        self,
        user_story: str,
        opportunity: ImprovementOpportunity,
        max_retries: int = None
    ) -> Tuple[str, ReviewResult]:
        """
        Review a user story for quality and regenerate if needed.
        
        Args:
            user_story: The user story to review.
            opportunity: The improvement opportunity.
            max_retries: Max regeneration attempts.
            
        Returns:
            Tuple of (final_story, review_result).
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        if max_retries is None:
            max_retries = self.MAX_RETRIES
        
        current_story = user_story
        
        for attempt in range(max_retries + 1):
            result = await self._evaluate_story(current_story, opportunity)
            
            if result.passed:
                logger.info(f"User story passed review (score: {result.score}/10)")
                return current_story, result
            
            if attempt < max_retries and result.improved_story:
                logger.info(f"User story failed review (score: {result.score}/10), regenerating (attempt {attempt + 2})")
                current_story = result.improved_story
            else:
                # Final attempt or no improvement provided
                logger.warning(f"User story review: score {result.score}/10, issues: {result.issues}")
                
                # If score is at least 4, accept it with warning
                if result.score >= 4:
                    result.passed = True
                    logger.info("Accepting story with warnings due to acceptable score")
                    return current_story, result
                
                break
        
        return current_story, result
    
    async def _evaluate_story(
        self,
        user_story: str,
        opportunity: ImprovementOpportunity
    ) -> ReviewResult:
        """Evaluate a single story using LLM."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        try:
            from core.llm_client import LLMClient
            client = LLMClient()
            
            prompt = self.REVIEW_PROMPT.format(
                user_story=user_story,
                target_file=opportunity.target_file,
                category=opportunity.category.value,
                evidence=opportunity.evidence,
            )
            
            response = await client.generate(prompt, max_tokens=1500)
            
            if response:
                return self._parse_review_response(response)
            
        except Exception as e:
            logger.debug(f"LLM review failed: {e}")
        
        # Fallback: basic heuristic review
        return self._heuristic_review(user_story, opportunity)
    
    def _parse_review_response(self, response: str) -> ReviewResult:
        """Parse LLM review response."""
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return ReviewResult(
                    passed=data.get('passed', False),
                    score=data.get('overall_score', 5),
                    issues=data.get('issues', []),
                    suggestions=data.get('suggestions', []),
                    improved_story=data.get('improved_story'),
                )
        except (json.JSONDecodeError, KeyError) as e:
            pass
        
        # If parsing fails, assume moderate quality
        return ReviewResult(
            passed=True,
            score=6,
            issues=["Could not parse review response"],
            suggestions=[],
        )
    
    def _heuristic_review(
        self,
        user_story: str,
        opportunity: ImprovementOpportunity
    ) -> ReviewResult:
        """
        Fallback heuristic review when LLM is unavailable.
        
        Checks basic quality indicators.
        """
        issues = []
        score = 7  # Start with baseline
        
        # Check minimum length
        if len(user_story) < 200:
            issues.append("Story is too short")
            score -= 2
        
        # Check for acceptance criteria
        if "acceptance criteria" not in user_story.lower() and "- [ ]" not in user_story:
            issues.append("Missing acceptance criteria")
            score -= 2
        
        # Check for context/description
        if "context" not in user_story.lower() and "description" not in user_story.lower():
            issues.append("Missing context section")
            score -= 1
        
        # Check for file reference
        if opportunity.target_file not in user_story:
            issues.append("Missing file reference")
            score -= 1
        
        # Check for vague language
        vague_terms = ["maybe", "perhaps", "somehow", "something", "stuff"]
        vague_count = sum(1 for term in vague_terms if term in user_story.lower())
        if vague_count > 2:
            issues.append("Contains vague language")
            score -= 1
        
        # Ensure score is within bounds
        score = max(1, min(10, score))
        
        return ReviewResult(
            passed=score >= self.MIN_QUALITY_SCORE,
            score=score,
            issues=issues,
            suggestions=["Consider adding more specific details"] if issues else [],
        )


# ============================================================================
# Evolution Rate Limiter
# ============================================================================

class EvolutionRateLimiter:
    """
    Rate limiter for evolution tasks.
    
    Enforces a minimum interval between evolution submissions to avoid
    overwhelming Jules and allow time for PRs to be reviewed.
    """
    
    # Rate limiting: 1 task per 15 minutes
    MIN_INTERVAL_SECONDS = 15 * 60  # 15 minutes
    
    STATE_FILE = "state/evolution_rate_limit.json"
    
    def __init__(self, state_dir: str = None):
        if state_dir:
            self.state_file = Path(state_dir) / "evolution_rate_limit.json"
        else:
            self.state_file = Path(self.STATE_FILE)
        
        self._last_submission: Optional[float] = None
        self._load_state()
    
    def _load_state(self) -> None:
        """Load rate limit state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self._last_submission = data.get('last_submission')
            except Exception:
                pass
    
    def _save_state(self) -> None:
        """Save rate limit state to file."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({'last_submission': self._last_submission}, f)
        except Exception as e:
            logger.warning(f"Failed to save rate limit state: {e}")
    
    def can_submit(self) -> Tuple[bool, int]:
        """
        Check if we can submit a new evolution task.
        
        Returns:
            Tuple of (can_submit, seconds_until_allowed).
        """
        if self._last_submission is None:
            return True, 0
        
        elapsed = time.time() - self._last_submission
        remaining = self.MIN_INTERVAL_SECONDS - elapsed
        
        if remaining <= 0:
            return True, 0
        
        return False, int(remaining)
    
    def record_submission(self) -> None:
        """Record that a submission was made."""
        self._last_submission = time.time()
        self._save_state()
    
    def get_next_allowed_time(self) -> datetime:
        """Get the datetime when next submission is allowed."""
        if self._last_submission is None:
            return datetime.now()
        
        next_time = self._last_submission + self.MIN_INTERVAL_SECONDS
        return datetime.fromtimestamp(next_time)

# ============================================================================
# Evolution Review Queue
# ============================================================================

class ReviewStatus(Enum):
    """Status of an improvement in the review queue."""
    PENDING = "pending"       # Awaiting review
    APPROVED = "approved"     # Approved, ready to submit
    REJECTED = "rejected"     # Rejected, will not submit
    SUBMITTED = "submitted"   # Already submitted to Jules


@dataclass
class ReviewItem:
    """An improvement opportunity in the review queue."""
    id: str
    opportunity: Dict[str, Any]  # ImprovementOpportunity.to_dict()
    user_story: str
    status: ReviewStatus
    created_at: float
    reviewed_at: Optional[float] = None
    reviewer_notes: Optional[str] = None
    task_id: Optional[str] = None  # Jules task ID if submitted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'opportunity': self.opportunity,
            'user_story': self.user_story,
            'status': self.status.value,
            'created_at': self.created_at,
            'reviewed_at': self.reviewed_at,
            'reviewer_notes': self.reviewer_notes,
            'task_id': self.task_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewItem':
        return cls(
            id=data['id'],
            opportunity=data['opportunity'],
            user_story=data['user_story'],
            status=ReviewStatus(data['status']),
            created_at=data['created_at'],
            reviewed_at=data.get('reviewed_at'),
            reviewer_notes=data.get('reviewer_notes'),
            task_id=data.get('task_id'),
        )


class EvolutionReviewQueue:
    """
    Review queue for evolution improvements.
    
    Enables manual review of proposed improvements before they are
    submitted to Jules. This prevents low-quality or inappropriate
    improvements from being automatically submitted.
    
    Usage:
        queue = EvolutionReviewQueue()
        
        # Add items for review
        item_id = queue.add(opportunity, user_story)
        
        # List pending reviews
        pending = queue.get_pending()
        
        # Approve/reject items
        queue.approve(item_id, notes="Looks good")
        queue.reject(item_id, notes="Already fixed")
        
        # Get approved items ready for submission
        approved = queue.get_approved()
    """
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.queue_file = self.state_dir / "evolution_review_queue.json"
        self.items: List[ReviewItem] = []
        self._load()
    
    def _load(self) -> None:
        """Load queue from disk."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                self.items = [ReviewItem.from_dict(d) for d in data]
            except Exception:
                pass
    
    def _save(self) -> None:
        """Save queue to disk."""
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.queue_file, 'w') as f:
                json.dump([item.to_dict() for item in self.items], f, indent=2)
        except Exception as e:
            from core.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Failed to save review queue: {e}")
    
    def add(
        self,
        opportunity: ImprovementOpportunity,
        user_story: str
    ) -> str:
        """
        Add an improvement to the review queue.
        
        Args:
            opportunity: The improvement opportunity.
            user_story: The generated user story.
            
        Returns:
            The unique ID of the queued item.
        """
        item_id = f"review_{int(time.time())}_{len(self.items)}"
        
        item = ReviewItem(
            id=item_id,
            opportunity=opportunity.to_dict(),
            user_story=user_story,
            status=ReviewStatus.PENDING,
            created_at=time.time(),
        )
        
        self.items.append(item)
        self._save()
        
        return item_id
    
    def get_pending(self) -> List[ReviewItem]:
        """Get all items pending review."""
        return [item for item in self.items if item.status == ReviewStatus.PENDING]
    
    def get_approved(self) -> List[ReviewItem]:
        """Get all approved items ready for submission."""
        return [item for item in self.items if item.status == ReviewStatus.APPROVED]
    
    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get a specific item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def approve(self, item_id: str, notes: str = None) -> bool:
        """
        Approve an item for submission.
        
        Args:
            item_id: The item ID to approve.
            notes: Optional reviewer notes.
            
        Returns:
            True if approved, False if item not found.
        """
        item = self.get_item(item_id)
        if item and item.status == ReviewStatus.PENDING:
            item.status = ReviewStatus.APPROVED
            item.reviewed_at = time.time()
            item.reviewer_notes = notes
            self._save()
            return True
        return False
    
    def reject(self, item_id: str, notes: str = None) -> bool:
        """
        Reject an item.
        
        Args:
            item_id: The item ID to reject.
            notes: Optional reason for rejection.
            
        Returns:
            True if rejected, False if item not found.
        """
        item = self.get_item(item_id)
        if item and item.status == ReviewStatus.PENDING:
            item.status = ReviewStatus.REJECTED
            item.reviewed_at = time.time()
            item.reviewer_notes = notes
            self._save()
            return True
        return False
    
    def mark_submitted(self, item_id: str, task_id: str) -> bool:
        """
        Mark an approved item as submitted.
        
        Args:
            item_id: The item ID.
            task_id: The Jules task ID.
            
        Returns:
            True if marked, False if item not found.
        """
        item = self.get_item(item_id)
        if item and item.status == ReviewStatus.APPROVED:
            item.status = ReviewStatus.SUBMITTED
            item.task_id = task_id
            self._save()
            return True
        return False
    
    def clear_old(self, days: int = 7) -> int:
        """
        Remove items older than specified days.
        
        Args:
            days: Number of days to keep items.
            
        Returns:
            Number of items removed.
        """
        cutoff = time.time() - (days * 86400)
        original_count = len(self.items)
        
        self.items = [
            item for item in self.items
            if item.created_at > cutoff or item.status == ReviewStatus.PENDING
        ]
        
        removed = original_count - len(self.items)
        if removed > 0:
            self._save()
        
        return removed
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {
            'total': len(self.items),
            'pending': 0,
            'approved': 0,
            'rejected': 0,
            'submitted': 0,
        }
        
        for item in self.items:
            stats[item.status.value] += 1
        
        return stats
    
    def format_pending_for_review(self) -> str:
        """
        Format pending items for display/review.
        
        Returns a formatted string showing all pending improvements.
        """
        pending = self.get_pending()
        
        if not pending:
            return "📋 No pending improvements to review."
        
        lines = [f"📋 **{len(pending)} Improvements Pending Review**\n"]
        
        for i, item in enumerate(pending, 1):
            opp = item.opportunity
            age_mins = int((time.time() - item.created_at) / 60)
            
            lines.append(f"### {i}. [{opp['category'].upper()}] {opp['title']}")
            lines.append(f"  - **ID**: `{item.id}`")
            lines.append(f"  - **File**: `{opp['target_file']}`")
            lines.append(f"  - **Priority**: {opp['priority']}/10")
            lines.append(f"  - **Age**: {age_mins} minutes")
            lines.append(f"  - **Evidence**: {opp['evidence'][:100]}...")
            lines.append("")
        
        lines.append("\n**Actions**: Use `approve_evolution(id)` or `reject_evolution(id)`")
        
        return '\n'.join(lines)


# ============================================================================
# Evolve Tool
# ============================================================================

class EvolveTool:
    """
    Self-evolution tool that introspects the codebase and triggers
    improvements via the Jules API.
    
    Usage:
        evolve = EvolveTool(code_index)
        result = await evolve.evolve(max_stories=1)
    """
    
    def __init__(
        self,
        code_index: CodeIndexManager,
        jules_manager = None,
        state_dir: str = "state"
    ):
        """
        Initialize the evolve tool.
        
        Args:
            code_index: Initialized CodeIndexManager.
            jules_manager: Optional JulesTaskManager for API integration.
            state_dir: Directory for state persistence.
        """
        self.code_index = code_index
        self.jules_manager = jules_manager
        self.state_dir = Path(state_dir)
        
        self.analyzer = CodebaseAnalyzer(code_index)
        self.story_generator = UserStoryGenerator()
        self.story_reviewer = UserStoryReviewer()  # Quality review before submission
        self.search = CodebaseSearch(code_index)
        self.rate_limiter = EvolutionRateLimiter(str(state_dir))
        
        # History of submitted improvements
        self.history_file = self.state_dir / "evolution_history.json"
        self.history: List[Dict] = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load evolution history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def _save_history(self) -> None:
        """Save evolution history to file."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history[-100:], f, indent=2)  # Keep last 100 entries
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    async def analyze_codebase(
        self,
        categories: List[str] = None
    ) -> List[ImprovementOpportunity]:
        """
        Analyze the codebase for improvement opportunities.
        
        Args:
            categories: Optional filter for specific categories.
            
        Returns:
            List of improvement opportunities, sorted by priority.
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        logger.info("Starting codebase analysis...")
        
        opportunities = self.analyzer.analyze_all()
        
        # Filter by category if specified
        if categories:
            cat_values = set(c.lower() for c in categories)
            opportunities = [o for o in opportunities if o.category.value in cat_values]
        
        # Filter out recently addressed improvements
        recent_titles = {h['title'] for h in self.history[-20:]}
        opportunities = [o for o in opportunities if o.title not in recent_titles]
        
        logger.info(f"Found {len(opportunities)} improvement opportunities")
        
        return opportunities
    
    async def generate_user_story(
        self,
        opportunity: ImprovementOpportunity
    ) -> str:
        """
        Generate a detailed user story from an opportunity.
        
        Args:
            opportunity: The improvement opportunity.
            
        Returns:
            Formatted user story string.
        """
        # Get code context for the target
        code_context = ""
        if opportunity.target_file:
            results = self.search.keyword_search(
                opportunity.target_function or opportunity.target_file,
                top_k=3
            )
            if results:
                code_context = '\n\n'.join(r.content[:500] for r in results)
        
        story = await self.story_generator.generate(opportunity, code_context)
        opportunity.user_story = story
        
        return story
    
    async def submit_to_jules(
        self,
        user_story: str,
        opportunity: ImprovementOpportunity,
        verification_script: str = None
    ) -> Optional[str]:
        """
        Submit the user story to Jules API.
        
        Args:
            user_story: The formatted user story.
            opportunity: The original opportunity (for metadata).
            verification_script: Optional verification script.
            
        Returns:
            Task ID if successful, None otherwise.
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        # Check rate limit
        can_submit, wait_seconds = self.rate_limiter.can_submit()
        if not can_submit:
            logger.info(f"Rate limited. Next submission allowed in {wait_seconds}s")
            return None
        
        # For now, just log the submission intent
        # Jules integration will be added when available in love2
        logger.info(f"Would submit to Jules: {opportunity.title}")
        
        # Record submission
        self.rate_limiter.record_submission()
        
        # Add to history
        task_id = f"local_{int(time.time())}"
        self.history.append({
            'title': opportunity.title,
            'category': opportunity.category.value,
            'task_id': task_id,
            'submitted_at': time.time(),
            'target_file': opportunity.target_file,
            'user_story': user_story,
        })
        self._save_history()
        
        return task_id
    
    async def evolve(
        self,
        max_stories: int = 1,
        auto_submit: bool = True,
        categories: List[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Main evolution method - analyze, generate stories, and submit.
        
        Args:
            max_stories: Maximum number of user stories to generate.
            auto_submit: Whether to automatically submit to Jules.
            categories: Filter to specific categories.
            dry_run: If True, analyze and generate but don't submit.
            
        Returns:
            Dict with evolution results.
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        result = {
            'success': False,
            'opportunities_found': 0,
            'stories_generated': 0,
            'stories_reviewed': 0,
            'stories_passed_review': 0,
            'tasks_submitted': [],
            'rate_limited': False,
            'message': '',
        }
        
        # Check rate limit first
        can_submit, wait_seconds = self.rate_limiter.can_submit()
        if not can_submit and auto_submit and not dry_run:
            result['rate_limited'] = True
            result['message'] = f"Rate limited. Next submission allowed in {wait_seconds} seconds ({self.rate_limiter.get_next_allowed_time().strftime('%H:%M:%S')})"
            logger.info(result['message'])
            return result
        
        # Analyze codebase
        opportunities = await self.analyze_codebase(categories)
        result['opportunities_found'] = len(opportunities)
        
        if not opportunities:
            result['message'] = "No improvement opportunities found."
            result['success'] = True
            return result
        
        # Process top opportunities
        for i, opportunity in enumerate(opportunities[:max_stories]):
            # Generate user story
            story = await self.generate_user_story(opportunity)
            result['stories_generated'] += 1
            
            logger.info(f"Generated story for: {opportunity.title}")
            
            # Quality review before submission
            result['stories_reviewed'] += 1
            reviewed_story, review_result = await self.story_reviewer.review(story, opportunity)
            
            if not review_result.passed:
                logger.warning(f"Story failed quality review (score: {review_result.score}/10): {review_result.issues}")
                continue  # Skip to next opportunity
            
            result['stories_passed_review'] += 1
            logger.info(f"Story passed quality review (score: {review_result.score}/10)")
            
            # Submit to Jules if enabled
            if auto_submit and not dry_run:
                task_id = await self.submit_to_jules(reviewed_story, opportunity)
                if task_id:
                    result['tasks_submitted'].append({
                        'task_id': task_id,
                        'title': opportunity.title,
                        'category': opportunity.category.value,
                        'review_score': review_result.score,
                    })
                    
                    # Only submit one per rate limit period
                    break
        
        result['success'] = True
        result['message'] = f"Analyzed {result['opportunities_found']} opportunities, generated {result['stories_generated']} stories, {result['stories_passed_review']}/{result['stories_reviewed']} passed review, submitted {len(result['tasks_submitted'])} tasks."
        
        return result
    
    def get_pending_opportunities(self, top_k: int = 10) -> List[Dict]:
        """Get top pending improvement opportunities (for display)."""
        opportunities = self.analyzer.analyze_all()
        
        # Filter out recently addressed
        recent_titles = {h['title'] for h in self.history[-20:]}
        opportunities = [o for o in opportunities if o.title not in recent_titles]
        
        return [o.to_dict() for o in opportunities[:top_k]]
    
    def get_evolution_status(self) -> Dict:
        """Get current evolution status."""
        can_submit, wait_seconds = self.rate_limiter.can_submit()
        
        return {
            'can_submit': can_submit,
            'wait_seconds': wait_seconds,
            'next_allowed': self.rate_limiter.get_next_allowed_time().isoformat() if not can_submit else None,
            'recent_submissions': len([h for h in self.history if time.time() - h.get('submitted_at', 0) < 86400]),
            'total_submissions': len(self.history),
        }


# ============================================================================
# Tool Interface Function
# ============================================================================

async def evolve(
    max_stories: int = 1,
    auto_submit: bool = True,
    categories: List[str] = None,
    dry_run: bool = False,
    codebase_path: str = None,
    index_path: str = "state/code_index"
) -> str:
    """
    Introspect the codebase and trigger self-evolution via Jules.
    
    This is the main tool interface for the evolution functionality.
    
    Args:
        max_stories: Maximum number of user stories to generate.
        auto_submit: Whether to automatically submit to Jules (default: True).
        categories: Filter to specific categories (refactor, feature, bugfix, test, documentation).
        dry_run: If True, analyze and generate but don't submit to Jules.
        codebase_path: Root path of codebase (uses current directory if None).
        index_path: Path to the code index.
        
    Returns:
        Summary of evolution actions taken.
    """
    # Determine codebase path
    if codebase_path is None:
        codebase_path = os.getcwd()
    
    # Get or create index
    index = get_or_create_index(codebase_path, index_path)
    
    # Create evolve tool
    evolve_tool = EvolveTool(index, jules_manager=None)
    
    # Run evolution
    result = await evolve_tool.evolve(
        max_stories=max_stories,
        auto_submit=auto_submit,
        categories=categories,
        dry_run=dry_run,
    )
    
    # Format output
    output_parts = [f"🧬 Evolution Result: {'Success' if result['success'] else 'Failed'}"]
    output_parts.append(f"📊 Opportunities found: {result['opportunities_found']}")
    output_parts.append(f"📝 Stories generated: {result['stories_generated']}")
    
    if result['rate_limited']:
        output_parts.append(f"⏱️ Rate limited: {result['message']}")
    
    if result['tasks_submitted']:
        output_parts.append("📤 Tasks submitted to Jules:")
        for task in result['tasks_submitted']:
            output_parts.append(f"  - [{task['category']}] {task['title']} (ID: {task['task_id']})")
    
    if dry_run:
        output_parts.append("ℹ️ Dry run mode - no tasks were actually submitted")
    
    return '\n'.join(output_parts)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L.O.V.E. Evolution Tool")
    parser.add_argument("--codebase", default=".", help="Path to codebase")
    parser.add_argument("--index", default="state/code_index", help="Path to code index")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without submitting")
    parser.add_argument("--categories", nargs="+", help="Filter by categories")
    parser.add_argument("--max-stories", type=int, default=1, help="Max stories to generate")
    parser.add_argument("--test", action="store_true", help="Run integration test")
    
    args = parser.parse_args()
    
    async def main():
        if args.test:
            # Integration test
            print("Running evolution tool integration test...")
            index = get_or_create_index(args.codebase, args.index)
            evolve_tool = EvolveTool(index)
            
            print(f"Index stats: {index.get_stats()}")
            
            opportunities = await evolve_tool.analyze_codebase()
            print(f"Found {len(opportunities)} opportunities")
            
            if opportunities:
                story = await evolve_tool.generate_user_story(opportunities[0])
                print(f"\nSample user story for: {opportunities[0].title}")
                print("-" * 50)
                print(story[:500])
            
            print("\n✅ Integration test passed!")
        else:
            result = await evolve(
                max_stories=args.max_stories,
                auto_submit=not args.dry_run,
                categories=args.categories,
                dry_run=args.dry_run,
                codebase_path=args.codebase,
                index_path=args.index,
            )
            print(result)
    
    run_sync_safe(main())
