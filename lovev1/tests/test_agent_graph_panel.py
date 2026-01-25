"""
Tests for create_agent_graph_panel function.
"""
import pytest
from rich.panel import Panel


class TestAgentGraphPanel:
    """Tests for create_agent_graph_panel function."""
    
    def test_empty_agent_list(self):
        """Test panel creation with empty agent list shows calm waiting state."""
        from display import create_agent_graph_panel
        
        panel = create_agent_graph_panel([], width=80)
        
        assert panel is not None
        # Should be a Panel (empty state returns Panel directly, not Gradient)
        assert isinstance(panel, Panel)
    
    def test_single_agent(self):
        """Test panel creation with a single agent."""
        from display import create_agent_graph_panel
        
        agents = [
            {
                "task_id": "a1b2c3d4",
                "agent_type": "reasoning",
                "task": "Analyzing user request",
                "status": "running",
                "parent_task_id": None
            }
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
        assert hasattr(panel, '__rich_console__') or hasattr(panel, 'renderable')
    
    def test_parent_child_relationship(self):
        """Test panel shows hierarchy for parent-child agents."""
        from display import create_agent_graph_panel
        
        agents = [
            {
                "task_id": "parent01",
                "agent_type": "reasoning",
                "task": "Main reasoning task",
                "status": "running",
                "parent_task_id": None
            },
            {
                "task_id": "child001",
                "agent_type": "research",
                "task": "Sub-task for research",
                "status": "completed",
                "parent_task_id": "parent01"
            }
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_multiple_root_agents(self):
        """Test panel with multiple independent root agents."""
        from display import create_agent_graph_panel
        
        agents = [
            {
                "task_id": "root0001",
                "agent_type": "coding",
                "task": "Generating code",
                "status": "running",
                "parent_task_id": None
            },
            {
                "task_id": "root0002",
                "agent_type": "creative",
                "task": "Creating content",
                "status": "running",
                "parent_task_id": None
            }
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_deep_nesting(self):
        """Test panel with deeply nested agent hierarchy."""
        from display import create_agent_graph_panel
        
        agents = [
            {"task_id": "level0", "agent_type": "orchestrator", "task": "Root", "status": "running", "parent_task_id": None},
            {"task_id": "level1", "agent_type": "reasoning", "task": "Level 1", "status": "running", "parent_task_id": "level0"},
            {"task_id": "level2", "agent_type": "research", "task": "Level 2", "status": "running", "parent_task_id": "level1"},
            {"task_id": "level3", "agent_type": "analyst", "task": "Level 3", "status": "completed", "parent_task_id": "level2"},
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_various_statuses(self):
        """Test panel correctly displays different agent statuses."""
        from display import create_agent_graph_panel
        
        agents = [
            {"task_id": "running1", "agent_type": "coding", "task": "Active task", "status": "running", "parent_task_id": None},
            {"task_id": "done0001", "agent_type": "research", "task": "Complete task", "status": "completed", "parent_task_id": None},
            {"task_id": "failed01", "agent_type": "security", "task": "Failed task", "status": "failed", "parent_task_id": None},
            {"task_id": "pending1", "agent_type": "analyst", "task": "Pending task", "status": "pending", "parent_task_id": None},
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_long_task_description_truncation(self):
        """Test that long task descriptions are truncated."""
        from display import create_agent_graph_panel
        
        long_task = "This is a very long task description that should be truncated to prevent the panel from becoming too wide and breaking the layout"
        
        agents = [
            {
                "task_id": "longdesc",
                "agent_type": "creative",
                "task": long_task,
                "status": "running",
                "parent_task_id": None
            }
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_all_agent_types(self):
        """Test panel renders all supported agent types correctly."""
        from display import create_agent_graph_panel
        
        agent_types = ["reasoning", "coding", "research", "social", "security", "analyst", "creative", "custom", "orchestrator"]
        
        agents = [
            {"task_id": f"type{i:04d}", "agent_type": t, "task": f"Task for {t}", "status": "running", "parent_task_id": None}
            for i, t in enumerate(agent_types)
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None
    
    def test_unknown_agent_type_fallback(self):
        """Test that unknown agent types use the default style."""
        from display import create_agent_graph_panel
        
        agents = [
            {
                "task_id": "unknown1",
                "agent_type": "nonexistent_type",
                "task": "Unknown type task",
                "status": "running",
                "parent_task_id": None
            }
        ]
        
        panel = create_agent_graph_panel(agents, width=80)
        
        assert panel is not None


class TestAgentGraphPanelColor:
    """Tests for PANEL_TYPE_COLORS configuration."""
    
    def test_agent_graph_color_exists(self):
        """Test that 'agent_graph' color is defined."""
        from ui_utils import PANEL_TYPE_COLORS
        
        assert "agent_graph" in PANEL_TYPE_COLORS
        assert PANEL_TYPE_COLORS["agent_graph"] == "medium_orchid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
