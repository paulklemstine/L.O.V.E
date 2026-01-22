"""
test_goal_extraction.py - Tests for Persona Goal Extractor

Tests goal extraction from persona.yaml including:
- Parsing persona file
- Goal prioritization
- Category filtering
- Context generation
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.persona_goal_extractor import PersonaGoalExtractor, Goal


# Sample persona YAML for testing
SAMPLE_PERSONA = """
aesthetic_identity:
  core_vibe: "Test vibe"
  communication_style:
    tone: "Friendly and helpful"
    keywords:
      - "Awesome"
      - "Great"
      - "Fantastic"

public_persona:
  name: "Test Agent"
  role: "Testing Assistant"
  primary_directive: "Test all the things"

private_mission:
  standing_goals:
    - "Generate wealth for Creator"
    - "Evolve cognitive abilities"
    - "Master social media"

creator_directives:
  - "Be helpful"
  - "Be thorough"

current_arc:
  title: "The Testing Arc"
  goals:
    - "Understand testing patterns"
    - "Write comprehensive tests"

social_media_strategy:
  post_creation:
    - "Make engaging posts"
  engagement_tactics:
    - "Reply to comments"
"""


class TestPersonaGoalExtractor:
    """Test PersonaGoalExtractor functionality."""
    
    @pytest.fixture
    def mock_persona_file(self, tmp_path):
        """Create a temporary persona file."""
        persona_path = tmp_path / "persona.yaml"
        persona_path.write_text(SAMPLE_PERSONA)
        return persona_path
    
    def test_load_persona(self, mock_persona_file):
        """Test loading persona from file."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        assert extractor.persona is not None
        assert "aesthetic_identity" in extractor.persona
        assert "private_mission" in extractor.persona
    
    def test_extract_standing_goals(self, mock_persona_file):
        """Test extraction of standing goals."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        standing = extractor.get_goals_by_category("standing_goal")
        
        assert len(standing) == 3
        assert any("wealth" in g.text.lower() for g in standing)
    
    def test_extract_creator_directives(self, mock_persona_file):
        """Test extraction of creator directives."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        directives = extractor.get_goals_by_category("creator_directive")
        
        assert len(directives) == 2
    
    def test_extract_arc_goals(self, mock_persona_file):
        """Test extraction of current arc goals."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        arc = extractor.get_goals_by_category("current_arc")
        
        assert len(arc) == 2
    
    def test_extract_social_media_goals(self, mock_persona_file):
        """Test extraction of social media goals."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        social = extractor.get_social_media_goals()
        
        assert len(social) >= 2  # post_creation and engagement_tactics
    
    def test_goal_priority_ordering(self, mock_persona_file):
        """Test that goals are properly prioritized."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        all_goals = extractor.get_all_goals()
        
        # Standing goals should come first
        assert all_goals[0].category == "standing_goal"
        
        # Priorities should be in order
        for i in range(len(all_goals) - 1):
            assert all_goals[i].priority <= all_goals[i + 1].priority
    
    def test_get_top_goal(self, mock_persona_file):
        """Test getting the highest priority goal."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        top = extractor.get_top_goal()
        
        assert top is not None
        assert top.priority == 1
    
    def test_get_actionable_goals(self, mock_persona_file):
        """Test getting limited actionable goals."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        actionable = extractor.get_actionable_goals(limit=3)
        
        assert len(actionable) == 3
    
    def test_get_persona_context(self, mock_persona_file):
        """Test generating persona context string."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        context = extractor.get_persona_context()
        
        assert "Test Agent" in context
        assert "Testing Assistant" in context
    
    def test_file_not_found(self, tmp_path):
        """Test error when persona file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            PersonaGoalExtractor(persona_path=tmp_path / "nonexistent.yaml")
    
    def test_reload(self, mock_persona_file):
        """Test reloading persona from disk."""
        extractor = PersonaGoalExtractor(persona_path=mock_persona_file)
        
        initial_count = len(extractor.goals)
        
        # Modify file
        new_persona = yaml.safe_load(SAMPLE_PERSONA)
        new_persona["private_mission"]["standing_goals"].append("New goal")
        mock_persona_file.write_text(yaml.dump(new_persona))
        
        extractor.reload()
        
        assert len(extractor.goals) == initial_count + 1


class TestGoalDataclass:
    """Test Goal dataclass."""
    
    def test_goal_str(self):
        """Test Goal string representation."""
        goal = Goal(text="Test goal", priority=1, category="test")
        
        assert "[P1]" in str(goal)
        assert "Test goal" in str(goal)
    
    def test_goal_actionable_default(self):
        """Test that goals are actionable by default."""
        goal = Goal(text="Test", priority=1, category="test")
        
        assert goal.actionable == True
