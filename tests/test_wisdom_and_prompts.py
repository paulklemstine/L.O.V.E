
import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

from core.memory.schemas import WisdomEntry
from core.prompt_manager import (
    update_prompt_registry,
    restore_prompts_from_backup,
    critique_prompt
)


# =============================================================================
# Story 2.1: Wisdom System Tests
# =============================================================================

class TestWisdomEntry:
    """Tests for the WisdomEntry model."""
    
    def test_wisdom_entry_creation(self):
        """Test creating a WisdomEntry with required fields."""
        wisdom = WisdomEntry(
            situation="Agent attempted to parse malformed JSON",
            action="Used try-except with fallback to regex extraction",
            outcome="Successfully recovered data from 95% of malformed responses",
            principle="Always implement fallback parsing for LLM outputs"
        )
        
        assert wisdom.situation == "Agent attempted to parse malformed JSON"
        assert wisdom.action == "Used try-except with fallback to regex extraction"
        assert wisdom.outcome == "Successfully recovered data from 95% of malformed responses"
        assert wisdom.principle == "Always implement fallback parsing for LLM outputs"
        assert wisdom.confidence == 0.8  # Default
        assert wisdom.source == "experience"  # Default
    
    def test_wisdom_entry_with_all_fields(self):
        """Test creating a WisdomEntry with all optional fields."""
        wisdom = WisdomEntry(
            situation="Tool execution timed out",
            action="Increased timeout from 30s to 60s",
            outcome="Timeout errors reduced by 80%",
            principle="Set generous timeouts for external API calls",
            confidence=0.95,
            source="success",
            tags=["performance", "reliability", "tooling"]
        )
        
        assert wisdom.confidence == 0.95
        assert wisdom.source == "success"
        assert "performance" in wisdom.tags
    
    def test_wisdom_to_prompt_format(self):
        """Test formatting wisdom for prompt injection."""
        wisdom = WisdomEntry(
            situation="Loop detected in reasoning",
            action="Switched to different tool",
            outcome="Successfully broke out of loop",
            principle="If stuck, try alternative approaches"
        )
        
        formatted = wisdom.to_prompt_format()
        
        assert "**Situation**:" in formatted
        assert "**Action**:" in formatted
        assert "**Outcome**:" in formatted
        assert "**Principle**:" in formatted
        assert "Loop detected" in formatted
    
    def test_wisdom_to_concise_format(self):
        """Test concise formatting for one-line wisdom."""
        wisdom = WisdomEntry(
            situation="Syntax error in generated code",
            action="Added preflight check",
            outcome="Syntax errors caught before commit",
            principle="Always validate code before committing"
        )
        
        concise = wisdom.to_concise_format()
        
        assert concise.startswith("When")
        assert "syntax error" in concise.lower()
    
    def test_wisdom_confidence_bounds(self):
        """Test that confidence is bounded between 0 and 1."""
        # Valid confidence
        wisdom = WisdomEntry(
            situation="Test",
            action="Test",
            outcome="Test",
            principle="Test",
            confidence=0.5
        )
        assert wisdom.confidence == 0.5
        
        # At bounds
        wisdom_low = WisdomEntry(
            situation="Test",
            action="Test", 
            outcome="Test",
            principle="Test",
            confidence=0.0
        )
        assert wisdom_low.confidence == 0.0
        
        wisdom_high = WisdomEntry(
            situation="Test",
            action="Test",
            outcome="Test", 
            principle="Test",
            confidence=1.0
        )
        assert wisdom_high.confidence == 1.0


# =============================================================================
# Story 2.2: Prompt Optimization Tests
# =============================================================================

class TestUpdatePromptRegistry:
    """Tests for the update_prompt_registry function."""
    
    @pytest.fixture
    def mock_prompts_env(self, tmp_path):
        """Set up a mock prompts.yaml environment."""
        prompts_dir = tmp_path / "core"
        prompts_dir.mkdir()
        
        prompts_content = {
            "test_prompt": "This is a test prompt.",
            "another_prompt": "Another prompt for testing."
        }
        
        prompts_file = prompts_dir / "prompts.yaml"
        import yaml
        with open(prompts_file, 'w') as f:
            yaml.dump(prompts_content, f)
        
        return prompts_dir
    
    def test_backup_is_created(self, tmp_path):
        """Test that a backup is created before modification."""
        import yaml
        
        # Create mock prompts.yaml
        prompts_content = {"test_prompt": "Original content"}
        prompts_path = tmp_path / "prompts.yaml"
        backup_path = tmp_path / "prompts.yaml.bak"
        
        with open(prompts_path, 'w') as f:
            yaml.dump(prompts_content, f)
        
        # Patch the file paths
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = update_prompt_registry(
                    "test_prompt",
                    "Updated content",
                    "Testing backup"
                )
        
        # Check backup was created
        assert os.path.exists(backup_path)
    
    def test_prompt_key_not_found(self, tmp_path):
        """Test error when prompt key doesn't exist."""
        import yaml
        
        prompts_content = {"existing_prompt": "Content"}
        prompts_path = tmp_path / "prompts.yaml"
        
        with open(prompts_path, 'w') as f:
            yaml.dump(prompts_content, f)
        
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = update_prompt_registry(
                    "nonexistent_key",
                    "New content",
                    "Testing"
                )
        
        assert result["success"] is False
        assert "not found" in result["message"]


class TestRestorePromptsFromBackup:
    """Tests for the restore_prompts_from_backup function."""
    
    def test_restore_when_no_backup(self, tmp_path):
        """Test error when no backup file exists."""
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = restore_prompts_from_backup()
        
        assert result["success"] is False
        assert "No backup file found" in result["message"]
    
    def test_restore_from_backup(self, tmp_path):
        """Test successful restore from backup."""
        import yaml
        
        # Create prompts.yaml (modified version)
        prompts_path = tmp_path / "prompts.yaml"
        backup_path = tmp_path / "prompts.yaml.bak"
        
        modified_content = {"test_prompt": "Modified content"}
        original_content = {"test_prompt": "Original content"}
        
        with open(prompts_path, 'w') as f:
            yaml.dump(modified_content, f)
        
        with open(backup_path, 'w') as f:
            yaml.dump(original_content, f)
        
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = restore_prompts_from_backup()
        
        assert result["success"] is True
        
        # Verify content was restored
        with open(prompts_path, 'r') as f:
            restored = yaml.safe_load(f)
        
        assert restored["test_prompt"] == "Original content"


class TestCritiquePrompt:
    """Tests for the critique_prompt function."""
    
    def test_critique_long_prompt(self, tmp_path):
        """Test that long prompts are flagged for compression."""
        import yaml
        
        prompts_path = tmp_path / "prompts.yaml"
        long_prompt = "A" * 3000  # Very long prompt
        
        with open(prompts_path, 'w') as f:
            yaml.dump({"long_prompt": long_prompt}, f)
        
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = critique_prompt("long_prompt")
        
        assert any("compression" in imp.lower() for imp in result["suggested_improvements"])
    
    def test_critique_missing_examples(self, tmp_path):
        """Test that prompts without examples are flagged."""
        import yaml
        
        prompts_path = tmp_path / "prompts.yaml"
        prompt_without_example = "Do the thing. Follow instructions."
        
        with open(prompts_path, 'w') as f:
            yaml.dump({"no_example_prompt": prompt_without_example}, f)
        
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = critique_prompt("no_example_prompt")
        
        assert any("example" in imp.lower() for imp in result["suggested_improvements"])
    
    def test_critique_nonexistent_prompt(self, tmp_path):
        """Test critiquing a prompt that doesn't exist."""
        import yaml
        
        prompts_path = tmp_path / "prompts.yaml"
        
        with open(prompts_path, 'w') as f:
            yaml.dump({"existing_prompt": "Content"}, f)
        
        with patch('core.prompt_manager.os.path.dirname', return_value=str(tmp_path)):
            with patch('core.prompt_manager.os.path.abspath', return_value=str(tmp_path / "prompt_manager.py")):
                result = critique_prompt("ghost_prompt")
        
        assert "not found" in result["analysis"]
