"""
Tests for Story 1.2: Hierarchical Goal Decomposition
"""
import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock


class TestDecomposedTask:
    """Tests for DecomposedTask dataclass."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        from core.goal_decomposer import DecomposedTask, TaskType
        
        task = DecomposedTask(
            task="Profile agent.py",
            task_type=TaskType.ANALYSIS,
            priority=1
        )
        
        assert task.task == "Profile agent.py"
        assert task.task_type == TaskType.ANALYSIS
        assert task.priority == 1
        assert task.children == []
    
    def test_task_to_dict(self):
        """Test serialization to dict."""
        from core.goal_decomposer import DecomposedTask, TaskType, ExecutionMode
        
        task = DecomposedTask(
            task="Test task",
            task_type=TaskType.IMPLEMENTATION,
            priority=2,
            execution_mode=ExecutionMode.PARALLEL
        )
        
        d = task.to_dict()
        assert d["task"] == "Test task"
        assert d["type"] == "implementation"
        assert d["priority"] == 2
        assert d["execution_mode"] == "parallel"
    
    def test_get_leaf_tasks_single(self):
        """Test leaf task retrieval with no children."""
        from core.goal_decomposer import DecomposedTask
        
        task = DecomposedTask(task="Leaf task")
        leaves = task.get_leaf_tasks()
        
        assert len(leaves) == 1
        assert leaves[0].task == "Leaf task"
    
    def test_get_leaf_tasks_nested(self):
        """Test leaf task retrieval with nested structure."""
        from core.goal_decomposer import DecomposedTask
        
        leaf1 = DecomposedTask(task="Leaf 1")
        leaf2 = DecomposedTask(task="Leaf 2")
        parent = DecomposedTask(task="Parent", children=[leaf1, leaf2])
        root = DecomposedTask(task="Root", children=[parent])
        
        leaves = root.get_leaf_tasks()
        assert len(leaves) == 2
        assert {l.task for l in leaves} == {"Leaf 1", "Leaf 2"}


class TestGoalTree:
    """Tests for GoalTree dataclass."""
    
    def test_tree_creation(self):
        """Test goal tree creation."""
        from core.goal_decomposer import GoalTree, DecomposedTask
        
        root = DecomposedTask(task="Main goal")
        tree = GoalTree(original_goal="Test goal", root=root)
        
        assert tree.original_goal == "Test goal"
        assert tree.root.task == "Main goal"
    
    def test_tree_to_json(self):
        """Test JSON serialization."""
        from core.goal_decomposer import GoalTree, DecomposedTask, TaskType
        
        child1 = DecomposedTask(task="Sub-task 1", task_type=TaskType.ANALYSIS, priority=1)
        child2 = DecomposedTask(task="Sub-task 2", task_type=TaskType.IMPLEMENTATION, priority=2)
        root = DecomposedTask(task="Improve performance", children=[child1, child2])
        tree = GoalTree(original_goal="Improve generic performance", root=root)
        
        json_str = tree.to_json()
        data = json.loads(json_str)
        
        assert data["goal"] == "Improve generic performance"
        assert len(data["sub_tasks"]) == 2
        assert data["sub_tasks"][0]["task"] == "Sub-task 1"


class TestGoalDecomposer:
    """Tests for GoalDecomposer class."""
    
    def test_match_pattern_performance(self):
        """Test pattern matching for performance goals."""
        from core.goal_decomposer import GoalDecomposer
        
        decomposer = GoalDecomposer()
        pattern = decomposer._match_pattern("Improve generic performance")
        
        assert pattern is not None
        assert len(pattern) > 0
        assert any("Profile" in task["task"] for task in pattern)
    
    def test_match_pattern_security(self):
        """Test pattern matching for security goals."""
        from core.goal_decomposer import GoalDecomposer
        
        decomposer = GoalDecomposer()
        pattern = decomposer._match_pattern("Improve application security")
        
        assert pattern is not None
        assert any("Audit" in task["task"] for task in pattern)
    
    def test_match_pattern_no_match(self):
        """Test pattern matching with no match."""
        from core.goal_decomposer import GoalDecomposer
        
        decomposer = GoalDecomposer()
        pattern = decomposer._match_pattern("Build a spaceship")
        
        assert pattern is None
    
    def test_build_tree_from_pattern(self):
        """Test building tree from matched pattern."""
        from core.goal_decomposer import GoalDecomposer
        
        decomposer = GoalDecomposer()
        pattern = [
            {"task": "Task 1", "type": "analysis"},
            {"task": "Task 2", "type": "implementation"},
        ]
        
        tree = decomposer._build_tree_from_pattern("Test goal", pattern)
        
        assert tree.original_goal == "Test goal"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].task == "Task 1"
    
    @pytest.mark.asyncio
    async def test_decompose_with_pattern_match(self):
        """Test decompose with a matching pattern."""
        from core.goal_decomposer import GoalDecomposer
        
        decomposer = GoalDecomposer()
        tree = await decomposer.decompose("Improve generic performance")
        
        assert tree.original_goal == "Improve generic performance"
        assert len(tree.root.children) > 0
    
    def test_create_todo_entries_no_manager(self):
        """Test TODO creation without task manager."""
        from core.goal_decomposer import GoalDecomposer, GoalTree, DecomposedTask
        
        decomposer = GoalDecomposer(love_task_manager=None)
        tree = GoalTree(
            original_goal="Test",
            root=DecomposedTask(task="Test task")
        )
        
        task_ids = decomposer.create_todo_entries(tree)
        assert task_ids == []
    
    def test_create_todo_entries_with_manager(self):
        """Test TODO creation with task manager."""
        from core.goal_decomposer import GoalDecomposer, GoalTree, DecomposedTask, TaskType
        
        mock_manager = MagicMock()
        mock_manager.add_task.return_value = "task-123"
        
        decomposer = GoalDecomposer(love_task_manager=mock_manager)
        leaf = DecomposedTask(task="Do something", task_type=TaskType.IMPLEMENTATION)
        tree = GoalTree(original_goal="Test", root=leaf)
        
        task_ids = decomposer.create_todo_entries(tree)
        
        assert len(task_ids) == 1
        assert "task-123" in task_ids
        mock_manager.add_task.assert_called_once()


class TestDecomposeGoalFunction:
    """Tests for decompose_goal convenience function."""
    
    @pytest.mark.asyncio
    async def test_decompose_goal_returns_json(self):
        """Test that decompose_goal returns valid JSON."""
        from core.goal_decomposer import decompose_goal
        
        result = await decompose_goal("Improve generic performance", create_todos=False)
        
        # Should be valid JSON
        data = json.loads(result)
        assert "goal" in data
        assert "sub_tasks" in data
    
    @pytest.mark.asyncio
    async def test_decompose_goal_acceptance_criteria(self):
        """Test the specific acceptance criteria from Story 1.2."""
        from core.goal_decomposer import decompose_goal
        
        result = await decompose_goal("Improve generic performance", create_todos=False)
        data = json.loads(result)
        
        # Check required structure
        assert data["goal"] == "Improve generic performance"
        assert isinstance(data["sub_tasks"], list)
        assert len(data["sub_tasks"]) >= 3
        
        # Check that expected sub-tasks are present
        sub_task_names = [t["task"].lower() for t in data["sub_tasks"]]
        assert any("profile" in name for name in sub_task_names)
        assert any("optim" in name for name in sub_task_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
