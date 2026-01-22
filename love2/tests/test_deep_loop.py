"""
test_deep_loop.py - Tests for the DeepLoop

Tests the main autonomous reasoning loop including:
- Initialization
- Single iteration execution
- Goal selection
- Graceful shutdown
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add love2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.deep_loop import DeepLoop
from core.memory_system import MemorySystem
from core.persona_goal_extractor import Goal


class TestDeepLoopInit:
    """Test DeepLoop initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder:
            
            mock_llm.return_value = Mock()
            mock_persona.return_value = Mock()
            mock_folder.return_value = Mock()
            
            loop = DeepLoop()
            
            assert loop.iteration == 0
            assert loop.running == False
            assert loop.current_goal is None
    
    def test_init_with_max_iterations(self):
        """Test initialization with max iterations."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder:
            
            mock_llm.return_value = Mock()
            mock_persona.return_value = Mock()
            mock_folder.return_value = Mock()
            
            loop = DeepLoop(max_iterations=5)
            
            assert loop.max_iterations == 5


class TestDeepLoopIteration:
    """Test single iteration behavior."""
    
    @pytest.fixture
    def mock_loop(self, tmp_path):
        """Create a mocked DeepLoop for testing."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder:
            
            # Mock LLM
            llm = Mock()
            llm.generate_json.return_value = {
                "thought": "I should post to Bluesky",
                "action": "bluesky_post",
                "action_input": {"text": "Hello world!"},
                "reasoning": "Posting helps achieve social media goal"
            }
            mock_llm.return_value = llm
            
            # Mock persona
            persona = Mock()
            persona.get_actionable_goals.return_value = [
                Goal(text="Post to social media", priority=1, category="social_media")
            ]
            persona.get_persona_context.return_value = "Test persona"
            mock_persona.return_value = persona
            
            # Mock folder
            folder = Mock()
            folder.should_fold.return_value = False
            mock_folder.return_value = folder
            
            # Create loop with temp state dir
            loop = DeepLoop()
            loop.memory = MemorySystem(state_dir=tmp_path)
            
            return loop
    
    def test_run_iteration_with_no_goals(self, tmp_path):
        """Test iteration when no goals are available."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder:
            
            mock_llm.return_value = Mock()
            
            persona = Mock()
            persona.get_actionable_goals.return_value = []
            mock_persona.return_value = persona
            
            mock_folder.return_value = Mock()
            
            loop = DeepLoop()
            loop.memory = MemorySystem(state_dir=tmp_path)
            
            result = loop.run_iteration()
            
            assert result == False
    
    def test_run_iteration_with_skip_action(self, tmp_path):
        """Test iteration when LLM returns skip action."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder:
            
            llm = Mock()
            llm.generate_json.return_value = {
                "thought": "Cannot proceed",
                "action": "skip",
                "action_input": {},
                "reasoning": "No suitable action available"
            }
            mock_llm.return_value = llm
            
            persona = Mock()
            persona.get_actionable_goals.return_value = [
                Goal(text="Test goal", priority=1, category="test")
            ]
            persona.get_persona_context.return_value = "Test"
            mock_persona.return_value = persona
            
            folder = Mock()
            folder.should_fold.return_value = False
            mock_folder.return_value = folder
            
            loop = DeepLoop()
            loop.memory = MemorySystem(state_dir=tmp_path)
            
            result = loop.run_iteration()
            
            assert result == False


class TestDeepLoopRun:
    """Test the main run loop."""
    
    def test_run_stops_at_max_iterations(self, tmp_path):
        """Test that loop stops at max_iterations."""
        with patch('core.deep_loop.get_llm_client') as mock_llm, \
             patch('core.deep_loop.get_persona_extractor') as mock_persona, \
             patch('core.deep_loop.get_memory_folder') as mock_folder, \
             patch('time.sleep'):
            
            llm = Mock()
            llm.generate_json.return_value = {
                "thought": "Skipping",
                "action": "skip",
                "action_input": {},
                "reasoning": "Test"
            }
            mock_llm.return_value = llm
            
            persona = Mock()
            persona.get_actionable_goals.return_value = [
                Goal(text="Test", priority=1, category="test")
            ]
            persona.get_persona_context.return_value = "Test"
            mock_persona.return_value = persona
            
            folder = Mock()
            folder.should_fold.return_value = False
            mock_folder.return_value = folder
            
            loop = DeepLoop(max_iterations=2, sleep_seconds=0)
            loop.memory = MemorySystem(state_dir=tmp_path)
            
            loop.run()
            
            assert loop.iteration == 2
            assert loop.running == False
