"""
Test suite for user story validator
"""

import pytest
from core.user_story_validator import (
    UserStoryValidator,
    generate_user_story_template,
    format_validation_error
)


class TestUserStoryValidator:
    """Tests for the UserStoryValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = UserStoryValidator()
    
    def test_reject_vague_multi_task_input(self):
        """Test that vague multi-task input is rejected."""
        bad_input = """
        Fix errors in talent_scout by defining TalentManager class, 
        fix research_and_evolve by resolving dependencies and errors, 
        optimize for CPU-only operation due to no GPU, and identify 
        new strategic opportunities
        """
        
        validation = self.validator.validate(bad_input)
        
        assert not validation.is_valid
        assert len(validation.errors) > 0
        assert any("multiple tasks" in error.lower() for error in validation.errors)
    
    def test_reject_too_short(self):
        """Test that too-short input is rejected."""
        short_input = "Fix the thing"
        
        validation = self.validator.validate(short_input)
        
        assert not validation.is_valid
        assert any("too short" in error.lower() for error in validation.errors)
    
    def test_reject_missing_sections(self):
        """Test that input missing required sections is rejected."""
        incomplete_input = """
        # Fix Something
        
        This needs to be fixed.
        """
        
        validation = self.validator.validate(incomplete_input)
        
        assert not validation.is_valid
        # Should be missing acceptance_criteria and technical_specification
        assert len(validation.errors) >= 2
    
    def test_accept_complete_user_story(self):
        """Test that a complete, proper user story is accepted."""
        good_input = """
# User Story: Fix TalentManager Import Error

**As a** system administrator
**I want** the talent_scout tool to execute without import errors
**So that** the AI can discover and analyze creative professionals

## Acceptance Criteria

- [ ] TalentManager class is properly imported in core/tools.py
- [ ] talent_scout function can instantiate TalentManager without errors
- [ ] All existing talent scout functionality remains intact
- [ ] Unit tests pass for talent scout operations

## Technical Specification

**File**: `core/tools.py`

**Change**: Update imports section on lines 37-41

```python
# BEFORE
from core.talent_utils import (
    talent_manager,
    public_profile_aggregator,
    intelligence_synthesizer
)

# AFTER
from core.talent_utils import (
    talent_manager,
    public_profile_aggregator,
    intelligence_synthesizer
)
from core.talent_utils.manager import TalentManager
```

The talent_scout function on line 798 should now work correctly:
```python
talent_manager = TalentManager()
```

## Dependencies

None - this is a straightforward import fix

## Testing Strategy

```python
async def test_talent_scout_import():
    from core.tools import talent_scout
    result = await talent_scout(keywords="AI art", platforms="bluesky")
    assert "Error" not in result
```
        """
        
        validation = self.validator.validate(good_input)
        
        # Should pass validation
        assert validation.is_valid
        # May have warnings but no errors
        assert len(validation.errors) == 0
    
    def test_detect_multiple_tasks_pattern1(self):
        """Test detection of 'and also' pattern."""
        multi_task = "Fix the import and also implement CPU support"
        
        assert self.validator._contains_multiple_tasks(multi_task)
    
    def test_detect_multiple_tasks_pattern2(self):
        """Test detection of comma-separated tasks."""
        multi_task = "Fix import, implement feature, optimize performance"
        
        assert self.validator._contains_multiple_tasks(multi_task)
    
    def test_detect_multiple_tasks_pattern3(self):
        """Test detection of numbered list."""
        multi_task = """
        1. Fix the import error
        2. Implement CPU support
        3. Add logging
        """
        
        assert self.validator._contains_multiple_tasks(multi_task)
    
    def test_single_task_not_flagged(self):
        """Test that single task is not flagged as multiple."""
        single_task = "Fix TalentManager import error in core/tools.py by adding the missing import statement"
        
        assert not self.validator._contains_multiple_tasks(single_task)
    
    def test_parse_user_story_sections(self):
        """Test parsing of user story sections."""
        story = """
# Test Story

**As a** developer
**I want** to test parsing
**So that** validation works

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Technical Specification

File: test.py
Change: Add import
        """
        
        parsed = self.validator._parse_user_story(story)
        
        assert 'title' in parsed
        assert 'as_a' in parsed
        assert 'i_want' in parsed
        assert 'so_that' in parsed
        assert 'acceptance_criteria' in parsed
        assert len(parsed['acceptance_criteria']) == 3
        assert 'technical_specification' in parsed
    
    def test_format_validation_error(self):
        """Test error message formatting."""
        validation = self.validator.validate("Too short")
        error_msg = format_validation_error(validation)
        
        assert "❌" in error_msg
        assert "ERRORS" in error_msg
        assert len(error_msg) > 100  # Should be detailed
    
    def test_generate_template(self):
        """Test template generation."""
        template = generate_user_story_template("Test feature")
        
        assert "# User Story: Test feature" in template
        assert "**As a**" in template
        assert "**I want**" in template
        assert "**So that**" in template
        assert "Acceptance Criteria" in template
        assert "Technical Specification" in template
        assert "```python" in template
    
    def test_warn_on_vague_language(self):
        """Test that vague language triggers warnings."""
        vague_input = """
# Improve Performance

**As a** user
**I want** to improve performance
**So that** things are faster

## Acceptance Criteria

- [ ] Performance is better
- [ ] System is optimized
- [ ] Everything runs faster

## Technical Specification

Optimize the code to make it run better and faster with improved performance.
        """
        
        validation = self.validator.validate(vague_input)
        
        # Should have warnings about vague language
        assert len(validation.warnings) > 0
        assert any("vague" in warning.lower() for warning in validation.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
