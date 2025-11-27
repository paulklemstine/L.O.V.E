"""
User Story Validator for L.O.V.E. Evolution System

This module validates that evolution requests follow proper user story format
with complete specifications necessary for implementation.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


async def expand_to_user_story(vague_input: str, **kwargs) -> str:
    """
    Uses an LLM to expand a vague evolution request into a detailed user story.
    
    Args:
        vague_input: The vague or incomplete evolution request
        **kwargs: Additional context (knowledge_base, deep_agent_instance, etc.)
        
    Returns:
        Properly formatted, detailed user story specification
    """
    from core.llm_api import run_llm
    import core.logging
    
    core.logging.log_event(f"[User Story Expander] Expanding vague input into detailed user story: {vague_input[:100]}...", "INFO")
    
    # Build context from the codebase
    context = _gather_codebase_context(vague_input)
    
    expansion_prompt = f"""You are an expert software architect and technical writer. Your task is to transform a vague evolution request into a detailed, implementable user story specification.

VAGUE INPUT:
---
{vague_input}
---

CODEBASE CONTEXT:
---
{context}
---

YOUR TASK:
Transform this into a SINGLE, focused user story following this exact format:

# User Story: [Specific, Clear Title]

## User Story Format

**As a** [specific role]  
**I want** [specific feature/fix with exact details]  
**So that** [clear business value]

## Acceptance Criteria

- [ ] [Specific, testable criterion 1 - include file names]
- [ ] [Specific, testable criterion 2 - include expected behavior]
- [ ] [Specific, testable criterion 3 - include verification method]
- [ ] [Add more as needed - minimum 3]

## Technical Specification

### Files to Modify

- `path/to/file.py` (lines X-Y) - [What will change]

### Implementation Details

**File**: `path/to/specific/file.py`

**Change**: [Specific description]
```python
# BEFORE (if modifying existing code)
existing_code_here()

# AFTER
new_code_here()
```

### Expected Behavior

[Describe exactly how the system should behave after this change]

## Dependencies

[List any prerequisites or dependencies, or "None" if standalone]

## Testing Strategy

```python
# Specific test case
async def test_the_change():
    # Test implementation
    assert expected_result
```

CRITICAL RULES:
1. Focus on ONE task only - if the input mentions multiple tasks, choose the MOST CRITICAL one
2. Be SPECIFIC - include exact file paths, line numbers, function names
3. Provide CODE EXAMPLES - show before/after code
4. Make it TESTABLE - each acceptance criterion must be verifiable
5. Include TECHNICAL DETAILS - minimum 200 characters in technical specification

If the input mentions multiple tasks like "Fix X, Y, and Z", create a user story for ONLY the first/most critical task and note the others as "Future Work" at the end.

Respond with ONLY the formatted user story, no additional commentary.
"""

    try:
        response_dict = await run_llm(
            expansion_prompt,
            purpose="user_story_expansion",
            is_source_code=True,
            deep_agent_instance=kwargs.get('deep_agent_instance')
        )
        
        expanded_story = response_dict.get("result", "")
        
        # Clean up any markdown code blocks
        if "```markdown" in expanded_story:
            expanded_story = expanded_story.split("```markdown")[1].split("```")[0].strip()
        elif "```" in expanded_story:
            # Remove any other code block wrappers
            parts = expanded_story.split("```")
            if len(parts) >= 3:
                expanded_story = parts[1].strip()
        
        core.logging.log_event(f"[User Story Expander] Successfully expanded to {len(expanded_story)} characters", "INFO")
        
        return expanded_story.strip()
        
    except Exception as e:
        core.logging.log_event(f"[User Story Expander] Failed to expand user story: {e}", "ERROR")
        # Return a basic template as fallback
        return generate_user_story_template(vague_input[:100])


def _gather_codebase_context(vague_input: str) -> str:
    """
    Gathers relevant context from the codebase based on the vague input.
    
    Args:
        vague_input: The vague input to analyze
        
    Returns:
        Context string with relevant file information
    """
    import os
    
    context_parts = []
    
    # Extract potential file/module names from the input
    potential_files = re.findall(r'(\w+(?:_\w+)*(?:\.py)?)', vague_input)
    
    # Common L.O.V.E. files to check
    common_files = [
        'core/tools.py',
        'core/talent_utils/manager.py',
        'core/researcher.py',
        'core/llm_api.py',
        'love.py'
    ]
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    for file_path in common_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            # Check if this file is mentioned in the input
            file_basename = os.path.basename(file_path)
            if any(term.lower() in file_basename.lower() for term in potential_files):
                try:
                    with open(full_path, 'r', errors='ignore') as f:
                        # Get first 50 lines for context
                        lines = f.readlines()[:50]
                        context_parts.append(f"File: {file_path}\n{''.join(lines)}\n")
                except Exception:
                    pass
    
    if not context_parts:
        context_parts.append("No specific file context found. Use your knowledge of the L.O.V.E. codebase structure.")
    
    return "\n".join(context_parts[:2])  # Limit to 2 files to avoid token overflow


@dataclass
class UserStoryValidation:
    """Result of user story validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class UserStoryValidator:
    """
    Validates that evolution requests are properly formatted user stories
    with complete implementation details.
    """
    
    # Required sections in a proper user story
    REQUIRED_SECTIONS = [
        "title",
        "as_a",
        "i_want",
        "so_that",
        "acceptance_criteria",
        "technical_specification"
    ]
    
    # Optional but recommended sections
    RECOMMENDED_SECTIONS = [
        "dependencies",
        "testing_strategy",
        "files_to_modify"
    ]
    
    def __init__(self):
        self.min_acceptance_criteria = 3
        self.min_technical_details_length = 100
        self.max_tasks_per_story = 1
    
    def validate(self, user_story: str) -> UserStoryValidation:
        """
        Validates a user story for completeness and proper format.
        
        Args:
            user_story: The user story text to validate
            
        Returns:
            UserStoryValidation object with results
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check if input is too short
        if len(user_story.strip()) < 50:
            errors.append(
                "User story is too short. A proper user story should include "
                "title, user perspective, acceptance criteria, and technical details."
            )
            return UserStoryValidation(False, errors, warnings, suggestions)
        
        # Parse the user story into sections
        parsed = self._parse_user_story(user_story)
        
        # Check for multiple tasks (anti-pattern)
        if self._contains_multiple_tasks(user_story):
            errors.append(
                "User story appears to contain multiple tasks. "
                "Please split into separate user stories, one per evolution request."
            )
            suggestions.append(
                "Example of multiple tasks: 'Fix X, implement Y, and optimize Z' "
                "Should be: Three separate user stories"
            )
        
        # Validate required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in parsed or not parsed[section]:
                errors.append(f"Missing required section: '{section}'")
                suggestions.append(self._get_section_example(section))
        
        # Validate acceptance criteria
        if "acceptance_criteria" in parsed:
            criteria_count = len(parsed["acceptance_criteria"])
            if criteria_count < self.min_acceptance_criteria:
                warnings.append(
                    f"Only {criteria_count} acceptance criteria found. "
                    f"Recommended minimum: {self.min_acceptance_criteria}"
                )
        
        # Validate technical specification depth
        if "technical_specification" in parsed:
            tech_spec = parsed["technical_specification"]
            if len(tech_spec) < self.min_technical_details_length:
                errors.append(
                    "Technical specification is too vague. Please include:\n"
                    "  - Specific files to modify\n"
                    "  - Exact functions/classes to change\n"
                    "  - Code examples or pseudocode\n"
                    "  - Expected behavior changes"
                )
        
        # Check for recommended sections
        for section in self.RECOMMENDED_SECTIONS:
            if section not in parsed or not parsed[section]:
                warnings.append(
                    f"Missing recommended section: '{section}'. "
                    f"This helps ensure successful implementation."
                )
        
        # Check for vague language
        vague_patterns = [
            r'\b(fix errors?|resolve issues?|improve|optimize|enhance)\b(?!\s+\w+\s+by)',
            r'\b(and|also|additionally)\b.*\b(fix|implement|add|create)\b'
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, user_story, re.IGNORECASE):
                warnings.append(
                    "User story contains vague language. Be specific about:\n"
                    "  - WHAT exactly needs to change\n"
                    "  - WHERE in the codebase\n"
                    "  - HOW it should be implemented\n"
                    "  - WHY this change is needed"
                )
                break
        
        is_valid = len(errors) == 0
        
        return UserStoryValidation(is_valid, errors, warnings, suggestions)
    
    def _parse_user_story(self, text: str) -> Dict[str, any]:
        """
        Attempts to parse a user story into its component sections.
        
        Returns:
            Dictionary of parsed sections
        """
        parsed = {}
        
        # Extract title (first line or ## Title section)
        title_match = re.search(r'^#+ (.+)$', text, re.MULTILINE)
        if title_match:
            parsed['title'] = title_match.group(1).strip()
        else:
            # Use first line as title
            first_line = text.split('\n')[0].strip()
            if first_line:
                parsed['title'] = first_line
        
        # Extract "As a" statement
        as_a_match = re.search(r'\*\*As a\*\*\s+(.+?)(?=\n|$)', text, re.IGNORECASE)
        if as_a_match:
            parsed['as_a'] = as_a_match.group(1).strip()
        
        # Extract "I want" statement
        i_want_match = re.search(r'\*\*I want\*\*\s+(.+?)(?=\n|$)', text, re.IGNORECASE)
        if i_want_match:
            parsed['i_want'] = i_want_match.group(1).strip()
        
        # Extract "So that" statement
        so_that_match = re.search(r'\*\*So that\*\*\s+(.+?)(?=\n|$)', text, re.IGNORECASE)
        if so_that_match:
            parsed['so_that'] = so_that_match.group(1).strip()
        
        # Extract acceptance criteria (look for checkboxes or numbered list)
        criteria = []
        criteria_section = re.search(
            r'(?:Acceptance Criteria|AC):?\s*\n((?:[-*\d.]\s*\[[ x]\].*\n?)+)',
            text,
            re.IGNORECASE | re.MULTILINE
        )
        if criteria_section:
            criteria_text = criteria_section.group(1)
            criteria = re.findall(r'[-*\d.]\s*\[[ x]\]\s*(.+)', criteria_text)
        parsed['acceptance_criteria'] = criteria
        
        # Extract technical specification
        tech_spec_match = re.search(
            r'(?:Technical Specification|Implementation):?\s*\n(.+?)(?=\n#{1,3}\s|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if tech_spec_match:
            parsed['technical_specification'] = tech_spec_match.group(1).strip()
        
        # Extract dependencies
        deps_match = re.search(
            r'(?:Dependencies|Requires):?\s*\n(.+?)(?=\n#{1,3}\s|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if deps_match:
            parsed['dependencies'] = deps_match.group(1).strip()
        
        # Extract testing strategy
        test_match = re.search(
            r'(?:Testing|Test Strategy):?\s*\n(.+?)(?=\n#{1,3}\s|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if test_match:
            parsed['testing_strategy'] = test_match.group(1).strip()
        
        # Extract files to modify
        files_match = re.search(
            r'(?:Files to Modify|Files):?\s*\n(.+?)(?=\n#{1,3}\s|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if files_match:
            parsed['files_to_modify'] = files_match.group(1).strip()
        
        return parsed
    
    def _contains_multiple_tasks(self, text: str) -> bool:
        """
        Detects if the user story contains multiple distinct tasks.
        
        Returns:
            True if multiple tasks detected
        """
        # Look for patterns indicating multiple tasks
        multi_task_patterns = [
            r'\b(and|also)\s+(fix|implement|add|create|remove|update|optimize)\b',
            r',\s+(fix|implement|add|create|remove|update|optimize)\b',
            r'\d+\.\s+(Fix|Implement|Add|Create|Remove|Update|Optimize)',
            r'[-*]\s+(Fix|Implement|Add|Create|Remove|Update|Optimize)'
        ]
        
        matches = 0
        for pattern in multi_task_patterns:
            matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # If we find multiple action verbs, likely multiple tasks
        return matches >= 2
    
    def _get_section_example(self, section: str) -> str:
        """
        Returns an example for a missing section.
        
        Args:
            section: The section name
            
        Returns:
            Example text for that section
        """
        examples = {
            "title": "Example: Fix TalentManager Import Error in core/tools.py",
            "as_a": "Example: **As a** system administrator",
            "i_want": "Example: **I want** the talent_scout tool to execute without import errors",
            "so_that": "Example: **So that** the AI can discover and analyze creative professionals",
            "acceptance_criteria": (
                "Example:\n"
                "- [ ] TalentManager class is properly imported\n"
                "- [ ] talent_scout function executes without errors\n"
                "- [ ] All existing functionality remains intact"
            ),
            "technical_specification": (
                "Example:\n"
                "**File**: `core/tools.py`\n"
                "**Change**: Add import on line 37:\n"
                "```python\n"
                "from core.talent_utils.manager import TalentManager\n"
                "```"
            ),
            "dependencies": "Example: None - this is a straightforward import fix",
            "testing_strategy": (
                "Example:\n"
                "```python\n"
                "async def test_talent_scout():\n"
                "    result = await talent_scout(keywords='AI art')\n"
                "    assert 'Error' not in result\n"
                "```"
            ),
            "files_to_modify": "Example: `core/tools.py` (line 37)"
        }
        
        return examples.get(section, f"Please add the '{section}' section")


def generate_user_story_template(brief_description: str = "") -> str:
    """
    Generates a user story template for the user to fill out.
    
    Args:
        brief_description: Optional brief description to include in template
        
    Returns:
        Formatted user story template
    """
    template = f"""# User Story: {brief_description or "[Title Here]"}

## User Story Format

**As a** [role/persona]  
**I want** [specific feature/fix]  
**So that** [business value/benefit]

## Acceptance Criteria

- [ ] [Specific, testable criterion 1]
- [ ] [Specific, testable criterion 2]
- [ ] [Specific, testable criterion 3]
- [ ] [Add more as needed]

## Technical Specification

### Files to Modify

- `path/to/file1.py` - [What changes]
- `path/to/file2.py` - [What changes]

### Implementation Details

**File**: `path/to/file.py`

**Change 1**: [Description]
```python
# Code example showing the change
# BEFORE
old_code_here()

# AFTER
new_code_here()
```

**Change 2**: [Description]
```python
# Another code example
```

### Expected Behavior

[Describe how the system should behave after this change]

## Dependencies

- [List any dependencies or prerequisites]
- [e.g., "Requires User Story #123 to be completed first"]

## Testing Strategy

```python
# Example test case
async def test_new_feature():
    result = await new_function()
    assert result == expected_value
```

## Additional Context

[Any other relevant information, links, or references]

---

## ⚠️ Important Guidelines

1. **ONE TASK ONLY**: This user story should describe a single, focused change
2. **BE SPECIFIC**: Include exact file names, line numbers, function names
3. **SHOW CODE**: Provide code examples, not just descriptions
4. **TESTABLE**: Each acceptance criterion must be verifiable
5. **COMPLETE**: Include all information needed for implementation

## ❌ Bad Example (Vague, Multiple Tasks)

"Fix errors in talent_scout by defining TalentManager class, fix research_and_evolve by resolving dependencies and errors, optimize for CPU-only operation due to no GPU, and identify new strategic opportunities"

## ✅ Good Example (Specific, Single Task)

"Fix TalentManager import error in core/tools.py by adding the missing import statement from core.talent_utils.manager, ensuring the talent_scout function can instantiate TalentManager without NameError"
"""
    
    return template


def format_validation_error(validation: UserStoryValidation) -> str:
    """
    Formats validation errors into a helpful error message.
    
    Args:
        validation: The validation result
        
    Returns:
        Formatted error message with guidance
    """
    if validation.is_valid:
        return "✅ User story validation passed!"
    
    message = ["❌ User Story Validation Failed\n"]
    message.append("=" * 60)
    
    if validation.errors:
        message.append("\n🚫 ERRORS (must fix):")
        for i, error in enumerate(validation.errors, 1):
            message.append(f"\n{i}. {error}")
    
    if validation.warnings:
        message.append("\n\n⚠️  WARNINGS (should address):")
        for i, warning in enumerate(validation.warnings, 1):
            message.append(f"\n{i}. {warning}")
    
    if validation.suggestions:
        message.append("\n\n💡 SUGGESTIONS:")
        for i, suggestion in enumerate(validation.suggestions, 1):
            message.append(f"\n{i}. {suggestion}")
    
    message.append("\n\n" + "=" * 60)
    message.append("\n\n📝 To generate a proper user story template, use:")
    message.append("\n   generate_user_story_template('Your brief description')")
    
    return "\n".join(message)
