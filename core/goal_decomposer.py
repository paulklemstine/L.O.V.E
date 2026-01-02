"""
Story 1.2: Hierarchical Goal Decomposition

This module converts high-level, vague user requests into structured
task trees that can be executed in parallel or sequence.
"""
import json
import asyncio
import uuid
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


class TaskStatus(Enum):
    """Status of a task in the execution lifecycle."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REFINEMENT = "needs_refinement"


@dataclass
class DecomposedTask:
    """
    Represents a single task in the decomposition tree.
    
    Attributes:
        task: Human-readable task description
        task_id: Unique identifier for tracking
        task_type: Category of the task
        status: Current execution status
        priority: Execution priority (lower = higher priority)
        depth: Nesting level in the tree (0 = root)
        execution_mode: How children should be executed
        children: Sub-tasks
        metadata: Additional context
    """
    task: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = TaskType.IMPLEMENTATION
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    depth: int = 0
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    children: List['DecomposedTask'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Maximum allowed nesting depth to prevent infinite recursion (class constant)
    MAX_DEPTH: int = field(default=5, init=False, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "depth": self.depth,
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
    
    def mark_completed(self) -> None:
        """Mark this task as completed."""
        self.status = TaskStatus.COMPLETED
    
    def mark_failed(self, error_msg: str = None) -> None:
        """Mark this task as failed with optional error message."""
        self.status = TaskStatus.FAILED
        if error_msg:
            self.metadata["error"] = error_msg
    
    def needs_refinement(self) -> bool:
        """Check if this task needs to be re-decomposed."""
        return self.status == TaskStatus.NEEDS_REFINEMENT


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
    
    def __init__(self, love_task_manager=None, memory_manager=None):
        """
        Initialize the GoalDecomposer.
        
        Args:
            love_task_manager: Reference to LoveTaskManager for TODO creation
            memory_manager: Reference to MemoryManager for contextual awareness
        """
        self.love_task_manager = love_task_manager
        self.memory_manager = memory_manager
        self._completed_task_ids: set = set()
    
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
    
    async def _llm_decompose(self, goal: str, current_depth: int = 0) -> GoalTree:
        """
        Uses LLM to decompose a goal into a recursive hierarchical structure.
        
        Args:
            goal: The goal to decompose
            current_depth: Current nesting depth (for recursive calls)
            
        Returns:
            GoalTree from LLM decomposition
        """
        from core.llm_api import run_llm
        
        # Enforce maximum depth to prevent infinite recursion
        max_depth = DecomposedTask.MAX_DEPTH
        if current_depth >= max_depth:
            log_event(f"Maximum decomposition depth ({max_depth}) reached for: {goal}")
            return self._create_leaf_goal_tree(goal, current_depth)
        
        prompt = f"""You are a hierarchical task decomposition expert. Break down the following goal into a RECURSIVE tree of sub-tasks.

Goal: "{goal}"

RULES:
1. Create 2-5 specific, actionable sub-tasks
2. If a sub-task is complex, include nested "children" sub-tasks (up to {max_depth - current_depth} more levels)
3. Simple, atomic tasks should have NO children
4. Each task must have: task, type, priority, and optionally children

Types: "analysis", "implementation", "verification", "research"
Priority: 1-5 (1 = highest)
Execution: "parallel" or "sequential"

Return ONLY valid JSON with this structure:
{{
  "task": "Root task description",
  "type": "analysis",
  "execution_mode": "sequential",
  "children": [
    {{
      "task": "Simple atomic task",
      "type": "implementation",
      "priority": 1
    }},
    {{
      "task": "Complex task requiring breakdown",
      "type": "analysis",
      "priority": 2,
      "execution_mode": "parallel",
      "children": [
        {{"task": "Sub-sub-task 1", "type": "research", "priority": 1}},
        {{"task": "Sub-sub-task 2", "type": "implementation", "priority": 2}}
      ]
    }}
  ]
}}"""

        try:
            response = await run_llm(prompt, purpose="goal_decomposition")
            result_text = response.get("result", "{}")
            
            # Extract JSON from response (handle both object and array)
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                task_json = json.loads(json_match.group())
            else:
                # Fall back to flat array format
                array_match = re.search(r'\[[\s\S]*\]', result_text)
                if array_match:
                    tasks_array = json.loads(array_match.group())
                    task_json = {"task": goal, "type": "analysis", "children": tasks_array}
                else:
                    task_json = {}
            
            # Parse recursive structure into DecomposedTask tree
            root = self._parse_recursive_tasks(task_json, goal, current_depth)
            
            # Filter out completed tasks if memory_manager is available
            if self.memory_manager:
                root = await self._filter_completed_tasks(root)
            
        except Exception as e:
            log_event(f"LLM decomposition failed: {e}, using fallback")
            root = self._create_fallback_root(goal, current_depth)
        
        return GoalTree(original_goal=goal, root=root)
    
    def _parse_recursive_tasks(self, task_json: Dict, fallback_task: str, depth: int = 0) -> DecomposedTask:
        """
        Recursively parses JSON into nested DecomposedTask structure.
        
        Args:
            task_json: JSON dict with task data and optional children
            fallback_task: Fallback task description if not in JSON
            depth: Current nesting depth
            
        Returns:
            DecomposedTask with nested children
        """
        task_desc = task_json.get("task", fallback_task)
        task_type_str = task_json.get("type", "implementation")
        priority = task_json.get("priority", depth + 1)
        exec_mode_str = task_json.get("execution_mode", "sequential")
        
        # Parse children recursively
        children_json = task_json.get("children", [])
        children = []
        for child_json in children_json:
            if isinstance(child_json, dict):
                child_task = self._parse_recursive_tasks(child_json, "Unknown subtask", depth + 1)
                children.append(child_task)
        
        return DecomposedTask(
            task=task_desc,
            task_type=TaskType(task_type_str) if task_type_str in [t.value for t in TaskType] else TaskType.IMPLEMENTATION,
            priority=priority,
            depth=depth,
            execution_mode=ExecutionMode(exec_mode_str) if exec_mode_str in [e.value for e in ExecutionMode] else ExecutionMode.SEQUENTIAL,
            children=children
        )
    
    def _create_leaf_goal_tree(self, goal: str, depth: int) -> GoalTree:
        """Creates a simple leaf GoalTree when max depth is reached."""
        root = DecomposedTask(
            task=goal,
            task_type=TaskType.IMPLEMENTATION,
            depth=depth,
            metadata={"max_depth_reached": True}
        )
        return GoalTree(original_goal=goal, root=root)
    
    def _create_fallback_root(self, goal: str, depth: int) -> DecomposedTask:
        """Creates fallback decomposition when LLM fails."""
        return DecomposedTask(
            task=goal,
            task_type=TaskType.ANALYSIS,
            depth=depth,
            execution_mode=ExecutionMode.SEQUENTIAL,
            children=[
                DecomposedTask(task=f"Analyze requirements for: {goal}", task_type=TaskType.ANALYSIS, priority=1, depth=depth+1),
                DecomposedTask(task=f"Implement solution for: {goal}", task_type=TaskType.IMPLEMENTATION, priority=2, depth=depth+1),
                DecomposedTask(task=f"Verify implementation of: {goal}", task_type=TaskType.VERIFICATION, priority=3, depth=depth+1),
            ]
        )
    
    async def _filter_completed_tasks(self, root: DecomposedTask) -> DecomposedTask:
        """
        Filters out tasks that have already been completed based on memory_manager.
        
        Args:
            root: Root task of the tree
            
        Returns:
            Filtered root task with completed children removed
        """
        if not self.memory_manager:
            return root
        
        # Query memory for completed task IDs
        try:
            working_memory = self.memory_manager.get_from_working_memory("completed_tasks")
            if working_memory:
                self._completed_task_ids.update(working_memory)
        except Exception as e:
            log_event(f"Failed to retrieve completed tasks from memory: {e}")
        
        # Recursively filter children
        filtered_children = []
        for child in root.children:
            if child.task_id not in self._completed_task_ids:
                filtered_child = await self._filter_completed_tasks(child)
                if filtered_child.children or not child.children:  # Keep if has remaining children or was a leaf
                    filtered_children.append(filtered_child)
        
        root.children = filtered_children
        return root
    
    async def refine_subtask(self, subtask: DecomposedTask) -> DecomposedTask:
        """
        Re-decomposes a subtask that was too vague or failed execution.
        Implements Story 1.2: Dynamic Subtask Refinement.
        
        Args:
            subtask: The subtask that needs refinement
            
        Returns:
            New DecomposedTask with more granular children
        """
        log_event(f"Refining subtask: {subtask.task}")
        subtask.status = TaskStatus.IN_PROGRESS
        
        # Use LLM to re-decompose at a deeper level
        refined_tree = await self._llm_decompose(subtask.task, current_depth=subtask.depth)
        
        # Transfer the refined children to the original subtask
        subtask.children = refined_tree.root.children
        subtask.execution_mode = refined_tree.root.execution_mode
        subtask.status = TaskStatus.PENDING
        
        return subtask
    
    def mark_task_completed(self, task: DecomposedTask) -> None:
        """
        Marks a task as completed and stores in memory for contextual awareness.
        
        Args:
            task: The completed task
        """
        task.mark_completed()
        self._completed_task_ids.add(task.task_id)
        
        # Persist to memory_manager if available
        if self.memory_manager:
            try:
                self.memory_manager.set_in_working_memory(
                    "completed_tasks", 
                    list(self._completed_task_ids)
                )
            except Exception as e:
                log_event(f"Failed to persist completed task to memory: {e}")
    
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
