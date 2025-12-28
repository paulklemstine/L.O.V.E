"""
Tests for Story 1.1: The "Mirror" Test (Self-Reflection Engine)
"""
import pytest
from unittest.mock import patch, MagicMock
import json


class TestReflectionReport:
    """Tests for ReflectionReport class."""
    
    def test_report_creation(self):
        """Test basic report creation."""
        from core.reflection_engine import ReflectionReport
        
        report = ReflectionReport(
            interaction_count=10,
            findings=[{"type": "Test", "description": "Test finding"}],
            improvements=["Improve something"],
            persona_diff={"test": ["value"]}
        )
        
        assert report.interaction_count == 10
        assert len(report.findings) == 1
        assert len(report.improvements) == 1
        assert report.generated_at is not None
    
    def test_report_to_dict(self):
        """Test report serialization to dict."""
        from core.reflection_engine import ReflectionReport
        
        report = ReflectionReport(
            interaction_count=5,
            findings=[],
            improvements=["Do X"]
        )
        
        d = report.to_dict()
        assert "generated_at" in d
        assert d["interaction_count"] == 5
        assert d["improvements"] == ["Do X"]
    
    def test_report_to_markdown(self):
        """Test report Markdown generation."""
        from core.reflection_engine import ReflectionReport
        
        report = ReflectionReport(
            interaction_count=10,
            findings=[{"type": "Tool Overuse", "description": "execute used 50%"}],
            improvements=["Diversify tools"],
            persona_diff={"creator_directives": ["new directive"]}
        )
        
        md = report.to_markdown()
        assert "# 🪞 Self-Reflection Report" in md
        assert "Tool Overuse" in md
        assert "Diversify tools" in md
        assert "creator_directives" in md


class TestReflectionEngine:
    """Tests for ReflectionEngine class."""
    
    def test_get_interaction_history_empty(self):
        """Test with empty history."""
        from core.reflection_engine import ReflectionEngine
        
        engine = ReflectionEngine(love_state={"autopilot_history": []})
        history = engine.get_interaction_history(10)
        
        assert history == []
    
    def test_get_interaction_history_partial(self):
        """Test with fewer interactions than requested."""
        from core.reflection_engine import ReflectionEngine
        
        mock_history = [
            {"command": "execute ls", "result": "success"},
            {"command": "read_file test.py", "result": "content"}
        ]
        engine = ReflectionEngine(love_state={"autopilot_history": mock_history})
        history = engine.get_interaction_history(10)
        
        assert len(history) == 2
    
    def test_analyze_interactions_tool_counting(self):
        """Test that tool usage is correctly counted."""
        from core.reflection_engine import ReflectionEngine
        
        mock_history = [
            {"command": "execute ls", "result": "success"},
            {"command": "execute pwd", "result": "success"},
            {"command": "execute whoami", "result": "success"},
            {"command": "read_file test.py", "result": "content"},
            {"command": "execute date", "result": "success"},
        ]
        engine = ReflectionEngine(love_state={"autopilot_history": mock_history})
        analysis = engine.analyze_interactions(5)
        
        assert analysis["tool_usage"]["execute"] == 4
        assert analysis["tool_usage"]["read_file"] == 1
        assert "execute" in [t["tool"] for t in analysis["overused_tools"]]
    
    def test_analyze_interactions_repetitive_phrases(self):
        """Test detection of repetitive commands."""
        from core.reflection_engine import ReflectionEngine
        
        mock_history = [
            {"command": "execute ls", "result": "success"},
            {"command": "execute ls", "result": "success"},
            {"command": "execute ls", "result": "success"},
        ]
        engine = ReflectionEngine(love_state={"autopilot_history": mock_history})
        analysis = engine.analyze_interactions(3)
        
        assert len(analysis["repetitive_phrases"]) > 0
        assert analysis["repetitive_phrases"][0]["count"] == 3
    
    def test_analyze_interactions_error_patterns(self):
        """Test detection of error patterns."""
        from core.reflection_engine import ReflectionEngine
        
        mock_history = [
            {"command": "fail_cmd", "result": "Error: something failed"},
            {"command": "fail_cmd", "result": "Error: something failed"},
        ]
        engine = ReflectionEngine(love_state={"autopilot_history": mock_history})
        analysis = engine.analyze_interactions(2)
        
        assert len(analysis["error_patterns"]) > 0
    
    def test_generate_improvements_overuse(self):
        """Test improvement suggestions for tool overuse."""
        from core.reflection_engine import ReflectionEngine
        
        engine = ReflectionEngine(love_state={})
        analysis = {
            "overused_tools": [{"tool": "execute", "percentage": 80}],
            "repetitive_phrases": [],
            "error_patterns": []
        }
        
        improvements = engine.generate_improvements(analysis)
        assert any("diversif" in imp.lower() for imp in improvements)
    
    def test_generate_improvements_no_issues(self):
        """Test that no issues results in positive message."""
        from core.reflection_engine import ReflectionEngine
        
        engine = ReflectionEngine(love_state={})
        analysis = {
            "overused_tools": [],
            "repetitive_phrases": [],
            "error_patterns": []
        }
        
        improvements = engine.generate_improvements(analysis)
        assert any("No significant issues" in imp for imp in improvements)
    
    def test_suggest_persona_update_generates_diff(self):
        """Test that persona update suggestions are generated."""
        from core.reflection_engine import ReflectionEngine
        
        with patch.object(ReflectionEngine, '_load_persona', return_value={"creator_directives": []}):
            engine = ReflectionEngine(love_state={})
            improvements = ["Consider diversifying tool usage. Tool 'execute' was used 80%."]
            
            diff = engine.suggest_persona_update(improvements)
            assert "creator_directives" in diff
    
    def test_generate_report_full_flow(self):
        """Test complete report generation flow."""
        from core.reflection_engine import ReflectionEngine
        
        mock_history = [
            {"command": "execute ls", "result": "success"},
            {"command": "execute ls", "result": "success"},
            {"command": "read_file x.py", "result": "content"},
        ]
        engine = ReflectionEngine(love_state={"autopilot_history": mock_history})
        
        report = engine.generate_report(3)
        
        assert report.interaction_count == 3
        assert isinstance(report.findings, list)
        assert isinstance(report.improvements, list)


class TestReflectFunction:
    """Tests for the reflect() convenience function."""
    
    def test_reflect_returns_markdown(self):
        """Test that reflect() returns a Markdown string."""
        from core.reflection_engine import reflect
        import core.shared_state as shared_state
        
        # Mock the shared state
        shared_state.love_state = {
            "autopilot_history": [
                {"command": "test_cmd", "result": "ok"}
            ]
        }
        
        result = reflect(count=1, save=False)
        
        assert isinstance(result, str)
        assert "# 🪞 Self-Reflection Report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
