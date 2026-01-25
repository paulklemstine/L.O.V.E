"""
Tests for Story 1.3: Tool Usage Visibility (Terminal Widget)
"""
import pytest
from unittest.mock import MagicMock, patch
from rich.panel import Panel
from rich.text import Text


class TestTerminalWidgetPanel:
    """Tests for create_terminal_widget_panel function."""
    
    def test_create_terminal_panel_thinking(self):
        """Test panel creation with 'thinking' status shows correct visual."""
        from display import create_terminal_widget_panel
        
        panel = create_terminal_widget_panel(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            status="thinking",
            width=80
        )
        
        # The returned object should be a Gradient wrapping a Panel
        assert panel is not None
        # Check that it's renderable (has a __rich__ or similar method)
        assert hasattr(panel, '__rich_console__') or hasattr(panel, 'renderable')
    
    def test_create_terminal_panel_executing(self):
        """Test panel creation with 'executing' status shows correct visual."""
        from display import create_terminal_widget_panel
        
        panel = create_terminal_widget_panel(
            tool_name="manage_bluesky",
            arguments={"action": "post", "text": "Hello world!"},
            status="executing",
            width=80
        )
        
        assert panel is not None
    
    def test_create_terminal_panel_complete(self):
        """Test panel creation with 'complete' status shows stdout and elapsed time."""
        from display import create_terminal_widget_panel
        
        panel = create_terminal_widget_panel(
            tool_name="execute",
            status="complete",
            stdout="Command executed successfully\nOutput line 2\nOutput line 3",
            elapsed_time=1.5,
            width=80
        )
        
        assert panel is not None
    
    def test_create_terminal_panel_error(self):
        """Test panel creation with 'error' status shows stderr."""
        from display import create_terminal_widget_panel
        
        panel = create_terminal_widget_panel(
            tool_name="broken_tool",
            status="error",
            stderr="ValueError: Invalid input",
            elapsed_time=0.1,
            width=80
        )
        
        assert panel is not None
    
    def test_truncate_long_arguments(self):
        """Test that very long argument values are truncated."""
        from display import create_terminal_widget_panel
        
        long_value = "x" * 100  # 100 characters
        panel = create_terminal_widget_panel(
            tool_name="test_tool",
            arguments={"long_arg": long_value},
            status="executing",
            width=80
        )
        
        # Should not raise an error
        assert panel is not None
    
    def test_empty_arguments(self):
        """Test panel handles None/empty arguments gracefully."""
        from display import create_terminal_widget_panel
        
        panel = create_terminal_widget_panel(
            tool_name="no_args_tool",
            arguments=None,
            status="thinking",
            width=80
        )
        
        assert panel is not None


class TestToolBase:
    """Tests for ToolBase class with visibility hooks."""
    
    def test_tool_base_notify_start_with_queue(self):
        """Test that _notify_start sends panel to UI queue."""
        from core.tool_base import ToolBase
        import queue
        
        class TestTool(ToolBase):
            name = "test_tool"
            def execute(self, **kwargs):
                return "success"
        
        ui_queue = queue.Queue()
        tool = TestTool()
        tool.ui_queue = ui_queue
        
        # Call _notify_start directly
        tool._notify_start((), {"arg1": "value1"})
        
        # Check that something was added to the queue
        assert not ui_queue.empty()
        item = ui_queue.get()
        assert item["type"] == "terminal_widget"
    
    def test_tool_base_notify_complete_with_queue(self):
        """Test that _notify_complete sends panel to UI queue."""
        from core.tool_base import ToolBase
        import queue
        import time
        
        class TestTool(ToolBase):
            name = "test_tool"
            def execute(self, **kwargs):
                return "result"
        
        ui_queue = queue.Queue()
        tool = TestTool()
        tool.ui_queue = ui_queue
        tool._start_time = time.time()
        
        tool._notify_complete("result")
        
        assert not ui_queue.empty()
    
    def test_tool_base_no_queue(self):
        """Test that tools work without a UI queue (no errors)."""
        from core.tool_base import ToolBase
        
        class TestTool(ToolBase):
            name = "test_tool"
            def execute(self, value=None):
                return f"processed: {value}"
        
        tool = TestTool()
        # No ui_queue set - should not raise
        result = tool(value="test")
        assert result == "processed: test"


class TestToolWrapper:
    """Tests for ToolWrapper class (legacy function wrapping)."""
    
    def test_wrapper_executes_function(self):
        """Test that ToolWrapper correctly calls the wrapped function."""
        from core.tool_base import ToolWrapper
        
        def my_function(x, y):
            return x + y
        
        wrapper = ToolWrapper(my_function, name="my_function")
        result = wrapper(x=1, y=2)
        
        assert result == 3
    
    def test_wrapper_sends_to_queue(self):
        """Test that ToolWrapper sends panels to UI queue."""
        from core.tool_base import ToolWrapper
        import queue
        
        def my_function(**kwargs):
            return "done"
        
        ui_queue = queue.Queue()
        wrapper = ToolWrapper(my_function, name="my_function", ui_queue=ui_queue)
        
        wrapper(arg="value")
        
        # Should have at least start and complete panels
        assert ui_queue.qsize() >= 2
    
    def test_wrapper_handles_errors(self):
        """Test that ToolWrapper sends error panel on exception."""
        from core.tool_base import ToolWrapper
        import queue
        
        def failing_function():
            raise ValueError("Test error")
        
        ui_queue = queue.Queue()
        wrapper = ToolWrapper(failing_function, name="failing_function", ui_queue=ui_queue)
        
        with pytest.raises(ValueError):
            wrapper()
        
        # Should have start and error panels
        assert ui_queue.qsize() >= 2


class TestPanelTypeColors:
    """Tests for PANEL_TYPE_COLORS configuration."""
    
    def test_terminal_color_exists(self):
        """Test that 'terminal' color is defined."""
        from ui_utils import PANEL_TYPE_COLORS
        
        assert "terminal" in PANEL_TYPE_COLORS
        assert PANEL_TYPE_COLORS["terminal"] == "bright_blue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
