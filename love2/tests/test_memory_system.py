"""
test_memory_system.py - Tests for the Memory System

Tests the brain-inspired memory system including:
- Episodic Memory CRUD
- Working Memory updates
- Tool Memory consolidation
- Persistence
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory_system import (
    MemorySystem,
    EpisodicMemory,
    EpisodicEvent,
    WorkingMemory,
    ToolMemory,
    ToolUsageStats
)


class TestEpisodicMemory:
    """Test Episodic Memory functionality."""
    
    def test_add_event(self):
        """Test adding events to episodic memory."""
        memory = EpisodicMemory()
        
        memory.add_event("goal_completed", "Finished task X")
        
        assert len(memory.events) == 1
        assert memory.events[0].event_type == "goal_completed"
        assert memory.events[0].summary == "Finished task X"
    
    def test_get_recent(self):
        """Test getting recent events."""
        memory = EpisodicMemory()
        
        for i in range(5):
            memory.add_event("test", f"Event {i}")
        
        recent = memory.get_recent(3)
        
        assert len(recent) == 3
        assert recent[-1].summary == "Event 4"
    
    def test_get_by_type(self):
        """Test filtering events by type."""
        memory = EpisodicMemory()
        
        memory.add_event("success", "Good thing 1")
        memory.add_event("error", "Bad thing")
        memory.add_event("success", "Good thing 2")
        
        successes = memory.get_by_type("success")
        
        assert len(successes) == 2
    
    def test_max_events_limit(self):
        """Test that old events are trimmed."""
        memory = EpisodicMemory(max_events=5)
        
        for i in range(10):
            memory.add_event("test", f"Event {i}")
        
        assert len(memory.events) == 5
        assert memory.events[0].summary == "Event 5"  # Oldest remaining
    
    def test_to_context_string(self):
        """Test context string formatting."""
        memory = EpisodicMemory()
        memory.add_event("test", "Something happened")
        
        context = memory.to_context_string()
        
        assert "Recent Events" in context
        assert "Something happened" in context


class TestWorkingMemory:
    """Test Working Memory functionality."""
    
    def test_set_goal(self):
        """Test setting current goal."""
        memory = WorkingMemory()
        
        memory.set_goal("Build a feature", sub_goals=["Design", "Implement"])
        
        assert memory.current_goal == "Build a feature"
        assert len(memory.sub_goals) == 2
    
    def test_set_plan(self):
        """Test setting plan."""
        memory = WorkingMemory()
        memory.set_goal("Test goal")
        
        memory.set_plan(["Step 1", "Step 2", "Step 3"])
        
        assert len(memory.plan) == 3
    
    def test_record_action(self):
        """Test recording actions."""
        memory = WorkingMemory()
        
        memory.record_action("do_thing", "success")
        
        assert memory.last_action == "do_thing"
        assert memory.last_result == "success"
        assert memory.iteration_count == 1
    
    def test_complete_sub_goal(self):
        """Test completing sub-goals."""
        memory = WorkingMemory()
        memory.set_goal("Main", sub_goals=["Sub1", "Sub2", "Sub3"])
        
        memory.complete_sub_goal("Sub2")
        
        assert "Sub2" not in memory.sub_goals
        assert len(memory.sub_goals) == 2


class TestToolMemory:
    """Test Tool Memory functionality."""
    
    def test_record_usage_success(self):
        """Test recording successful tool usage."""
        memory = ToolMemory()
        
        memory.record_usage("bluesky_post", success=True, execution_time_ms=100)
        
        assert "bluesky_post" in memory.tool_usage
        stats = memory.tool_usage["bluesky_post"]
        assert stats.success_count == 1
        assert stats.failure_count == 0
    
    def test_record_usage_failure(self):
        """Test recording failed tool usage."""
        memory = ToolMemory()
        
        memory.record_usage(
            "bluesky_post", 
            success=False, 
            execution_time_ms=50,
            error="Connection timeout"
        )
        
        stats = memory.tool_usage["bluesky_post"]
        assert stats.failure_count == 1
        assert stats.last_error == "Connection timeout"
    
    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ToolUsageStats(success_count=8, failure_count=2)
        
        assert stats.success_rate == 0.8
    
    def test_get_reliable_tools(self):
        """Test getting reliable tools."""
        memory = ToolMemory()
        
        # Reliable tool
        for _ in range(9):
            memory.record_usage("good_tool", success=True, execution_time_ms=10)
        memory.record_usage("good_tool", success=False, execution_time_ms=10)
        
        # Unreliable tool
        for _ in range(5):
            memory.record_usage("bad_tool", success=True, execution_time_ms=10)
            memory.record_usage("bad_tool", success=False, execution_time_ms=10)
        
        reliable = memory.get_reliable_tools(min_success_rate=0.8)
        
        assert "good_tool" in reliable
        assert "bad_tool" not in reliable


class TestMemorySystemPersistence:
    """Test MemorySystem persistence."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading memory state."""
        # Create and populate memory
        memory1 = MemorySystem(state_dir=tmp_path)
        memory1.episodic.add_event("test", "Event 1")
        memory1.working.set_goal("Goal 1")
        memory1.tool.record_usage("tool1", success=True, execution_time_ms=100)
        memory1.save()
        
        # Create new instance and verify loaded state
        memory2 = MemorySystem(state_dir=tmp_path)
        
        assert len(memory2.episodic.events) == 1
        assert memory2.working.current_goal == "Goal 1"
        assert "tool1" in memory2.tool.tool_usage
    
    def test_record_action_persists(self, tmp_path):
        """Test that record_action saves state."""
        memory = MemorySystem(state_dir=tmp_path)
        
        memory.record_action(
            tool_name="test_tool",
            action="do_thing",
            result="done",
            success=True,
            time_ms=50
        )
        
        # Check file was written
        assert (tmp_path / "working_memory.json").exists()
        assert (tmp_path / "tool_memory.json").exists()
        assert (tmp_path / "episodic_memory.json").exists()
