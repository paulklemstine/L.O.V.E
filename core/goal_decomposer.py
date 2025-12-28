"""
Story 1.2: Hierarchical Goal Decomposition

This module converts high-level, vague user requests into structured
task trees that can be executed in parallel or sequence.
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from core.logging import log_event


class TaskType(Enum):
    """Types of tasks in the decomposition tree."""
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    RESEARCH = "research"


class ExecutionMode(Enum):
    """How child tasks should be executed."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class DecomposedTask:
    """
    Represents a single task in the decomposition tree.
    
    Attributes:
        task: Human-readable task description
        task_type: Category of the task
        priority: Execution priority (lower = higher priority)
        execution_mode: How children should be executed
        children: Sub-tasks
        metadata: Additional context
    """
    task: str
    task_type: TaskType = TaskType.IMPLEMENTATION
    priority: int = 1
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    children: List['DecomposedTask'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task": self.task,
            "type": self.task_type.value,
            "priority": self.priority,
            "execution_mode": self.execution_mode.value,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    def get_leaf_tasks(self) -> List['DecomposedTask']:
        """Returns all leaf nodes (executable tasks) in the tree."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_tasks())
        return leaves


@dataclass
class GoalTree:
    """
    Represents the complete hierarchical decomposition of a goal.
    
    Attributes:
        original_goal: The original vague input
        root: Root task of the decomposition
    """
    original_goal: str
    root: DecomposedTask
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "goal": self.original_goal,
            "sub_tasks": [child.to_dict() for child in self.root.children] if self.root.children else [self.root.to_dict()]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_all_leaf_tasks(self) -> List[DecomposedTask]:
        """Returns all executable leaf tasks."""
        return self.root.get_leaf_tasks()


class GoalDecomposer:
    """
    Converts high-level goals into actionable task trees.
    
    Uses LLM-based reasoning to break down vague requests into
    specific, testable sub-tasks.
    """
    
    # Common goal patterns and their typical decompositions
    DECOMPOSITION_PATTERNS = {
        "performance": [
            {"task": "Profile the codebase to identify bottlenecks", "type": "analysis"},
            {"task": "Optimize slow database queries", "type": "implementation"},
            {"task": "Add caching for frequently accessed data", "type": "implementation"},
            {"task": "Minify and compress static assets", "type": "implementation"},
            {"task": "Verify performance improvements with benchmarks", "type": "verification"},
        ],
        "security": [
            {"task": "Audit authentication and authorization logic", "type": "analysis"},
            {"task": "Scan for known vulnerabilities in dependencies", "type": "analysis"},
            {"task": "Implement input validation and sanitization", "type": "implementation"},
            {"task": "Add rate limiting to API endpoints", "type": "implementation"},
            {"task": "Run security test suite", "type": "verification"},
        ],
        "testing": [
            {"task": "Analyze current test coverage", "type": "analysis"},
            {"task": "Identify untested critical paths", "type": "analysis"},
            {"task": "Write unit tests for core modules", "type": "implementation"},
            {"task": "Add integration tests for API endpoints", "type": "implementation"},
            {"task": "Verify test suite passes", "type": "verification"},
        ],
        "documentation": [
            {"task": "Audit existing documentation for gaps", "type": "analysis"},
            {"task": "Document public APIs and interfaces", "type": "implementation"},
            {"task": "Create usage examples and tutorials", "type": "implementation"},
            {"task": "Update README and CONTRIBUTING files", "type": "implementation"},
        ],
    }
    
    def __init__(self, love_task_manager=None):
        """
        Initialize the GoalDecomposer.
        
        Args:
            love_task_manager: Reference to LoveTaskManager for TODO creation
        """
        self.love_task_manager = love_task_manager
    
    async def decompose(self, vague_goal: str) -> GoalTree:
        """
        Decomposes a vague goal into a structured task tree.
        
        Args:
            vague_goal: High-level goal like "Improve generic performance"
            
        Returns:
            GoalTree with hierarchical sub-tasks
        """
        log_event(f"Decomposing goal: {vague_goal}")
        
        # Try pattern matching first for common goals
        matched_pattern = self._match_pattern(vague_goal)
        if matched_pattern:
            return self._build_tree_from_pattern(vague_goal, matched_pattern)
        
        # Fall back to LLM-based decomposition
        return await self._llm_decompose(vague_goal)
    
    def _match_pattern(self, goal: str) -> Optional[List[Dict[str, str]]]:
        """
        Attempts to match goal against known patterns.
        
        Args:
            goal: The goal string
            
        Returns:
            Pattern tasks if matched, None otherwise
        """
        goal_lower = goal.lower()
        for pattern_key, tasks in self.DECOMPOSITION_PATTERNS.items():
            if pattern_key in goal_lower:
                return tasks
        return None
    
    def _build_tree_from_pattern(self, goal: str, pattern: List[Dict[str, str]]) -> GoalTree:
        """
        Builds a GoalTree from a matched pattern.
        
        Args:
            goal: Original goal string
            pattern: List of task dicts from pattern
            
        Returns:
            Constructed GoalTree
        """
        children = [
            DecomposedTask(
                task=task_info["task"],
                task_type=TaskType(task_info["type"]),
                priority=i + 1
            )
            for i, task_info in enumerate(pattern)
        ]
        
        root = DecomposedTask(
            task=goal,
            task_type=TaskType.ANALYSIS,
            children=children,
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        return GoalTree(original_goal=goal, root=root)
    
    async def _llm_decompose(self, goal: str) -> GoalTree:
        """
        Uses LLM to decompose a goal not matching known patterns.
        
        Args:
            goal: The goal to decompose
            
        Returns:
            GoalTree from LLM decomposition
        """
        from core.llm_api import run_llm
        
        prompt = f"""You are a task decomposition expert. Break down the following vague goal into 3-5 specific, actionable sub-tasks.

Goal: "{goal}"

For each sub-task, provide:
- task: A clear, actionable description
- type: One of "analysis", "implementation", "verification", "research"
- priority: A number 1-5 (1 is highest priority)

Return ONLY a valid JSON array with no additional text:
[
  {{"task": "...", "type": "...", "priority": 1}},
  ...
]"""

        try:
            response = await run_llm(prompt, purpose="goal_decomposition")
            result_text = response.get("result", "[]")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', result_text)
            if json_match:
                tasks_json = json.loads(json_match.group())
            else:
                tasks_json = []
            
            children = [
                DecomposedTask(
                    task=task_info.get("task", "Unknown task"),
                    task_type=TaskType(task_info.get("type", "implementation")),
                    priority=task_info.get("priority", 5)
                )
                for task_info in tasks_json
            ]
            
            # Sort by priority
            children.sort(key=lambda t: t.priority)
            
        except Exception as e:
            log_event(f"LLM decomposition failed: {e}, using fallback")
            # Fallback to generic tasks
            children = [
                DecomposedTask(task=f"Analyze requirements for: {goal}", task_type=TaskType.ANALYSIS, priority=1),
                DecomposedTask(task=f"Implement solution for: {goal}", task_type=TaskType.IMPLEMENTATION, priority=2),
                DecomposedTask(task=f"Verify implementation of: {goal}", task_type=TaskType.VERIFICATION, priority=3),
            ]
        
        root = DecomposedTask(
            task=goal,
            task_type=TaskType.ANALYSIS,
            children=children,
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        return GoalTree(original_goal=goal, root=root)
    
    def create_todo_entries(self, goal_tree: GoalTree) -> List[str]:
        """
        Creates TODO entries in LoveTaskManager for each leaf task.
        
        Args:
            goal_tree: The decomposed goal tree
            
        Returns:
            List of created task IDs
        """
        if not self.love_task_manager:
            log_event("No LoveTaskManager available for TODO creation")
            return []
        
        task_ids = []
        leaf_tasks = goal_tree.get_all_leaf_tasks()
        
        for leaf in leaf_tasks:
            try:
                task_id = self.love_task_manager.add_task(
                    f"[{leaf.task_type.value.upper()}] {leaf.task}"
                )
                if task_id:
                    task_ids.append(task_id)
            except Exception as e:
                log_event(f"Failed to create TODO for task '{leaf.task}': {e}")
        
        log_event(f"Created {len(task_ids)} TODO entries from goal decomposition")
        return task_ids


# Convenience function for tool invocation
async def decompose_goal(goal: str, create_todos: bool = True) -> str:
    """
    Decomposes a vague goal into specific sub-tasks.
    
    Args:
        goal: The high-level goal to decompose
        create_todos: Whether to create TODO entries for each task
        
    Returns:
        JSON string of the decomposed task tree
    """
    import core.shared_state as shared_state
    
    # Get task manager if available
    task_manager = getattr(shared_state, 'love_task_manager', None)
    
    decomposer = GoalDecomposer(love_task_manager=task_manager)
    goal_tree = await decomposer.decompose(goal)
    
    if create_todos and task_manager:
        task_ids = decomposer.create_todo_entries(goal_tree)
        log_event(f"Created {len(task_ids)} TODO entries")
    
    return goal_tree.to_json()
