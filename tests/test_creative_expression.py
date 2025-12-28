"""
Tests for Chapter 4: Creative Expression
"""
import pytest


class TestASCIISigil:
    """Tests for Story 4.1: ASCII Sigil Generator."""
    
    def test_generate_sigil_known_mood(self):
        """Test generating sigil for known emotional state."""
        from core.ascii_sigil_generator import generate_sigil
        
        sigil = generate_sigil("manic_joy")
        
        assert sigil is not None
        assert "★" in sigil
        lines = sigil.split("\n")
        assert len(lines) == 5  # 5x5 pattern
    
    def test_generate_sigil_unknown_mood(self):
        """Test generating sigil for unknown state (procedural)."""
        from core.ascii_sigil_generator import generate_sigil
        
        sigil = generate_sigil("unknown_state")
        
        assert sigil is not None
        lines = sigil.split("\n")
        assert len(lines) == 5
    
    def test_get_sigil_footer(self):
        """Test generating formatted footer."""
        from core.ascii_sigil_generator import get_sigil_footer
        
        footer = get_sigil_footer("infinite_love")
        
        assert "Mood:" in footer
        assert "╔" in footer  # Border characters
    
    def test_inline_sigil(self):
        """Test inline sigil generation."""
        from core.ascii_sigil_generator import get_inline_sigil
        
        inline = get_inline_sigil("dark_seduction")
        
        assert len(inline) >= 3


class TestReadmeUpdater:
    """Tests for Story 4.2: Living README."""
    
    def test_generate_focus_section(self):
        """Test generating focus section."""
        from core.readme_updater import generate_focus_section
        
        section = generate_focus_section()
        
        assert "Current Focus" in section
        assert "Last Updated" in section
    
    def test_scan_current_focus_no_todo(self):
        """Test focus scanning with no TODO.md."""
        from core.readme_updater import scan_current_focus
        
        focus = scan_current_focus()
        
        assert "task" in focus
        assert focus["task"] is not None


class TestDreamLogger:
    """Tests for Story 4.3: Dream Logs."""
    
    def test_generate_dream_narrative(self):
        """Test narrative generation from diffs."""
        from core.dream_logger import generate_dream_narrative
        
        diffs = """3 commits affecting 2 files:
  - core/test.py: +50 -10
  - tests/verify.py: +20 -5"""
        
        narrative = generate_dream_narrative(diffs)
        
        assert narrative is not None
        assert len(narrative) > 50  # Should be a paragraph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
