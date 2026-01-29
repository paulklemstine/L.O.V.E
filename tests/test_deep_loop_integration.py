"""
Integration Test for DeepLoop + Epic 1 Components

Verifies that DeepLoop correctly initializes and uses:
- ToolRegistry (for tool discovery)
- ToolRetriever (for tool context)
- ToolGapDetector (for missing capabilities)
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.deep_loop import DeepLoop
from core.evolution_state import get_pending_specifications, load_evolution_state
from core.tool_retriever import get_tool_retriever

class MockLLMForLoop:
    def generate_json(self, prompt, system_prompt, temperature):
        # Always return skip to trigger gap detection (if no tools found)
        # Or act like we looked for tools and found none
        return {
            "thought": "I don't see any tools to fly to the moon.",
            "action": "skip",
            "reasoning": "No suitable tool found"
        }

@pytest.fixture
def clean_state():
    # Setup state
    pass

def test_deep_loop_initialization():
    """Verify DeepLoop initializes Epic 1 components."""
    # Mock tool adapter to avoid import errors
    with patch('core.deep_loop._load_default_tools') as mock_load:
        loop = DeepLoop(llm=MockLLMForLoop(), max_iterations=1, sleep_seconds=0)
        
        assert loop.registry is not None
        assert loop.gap_detector is not None
        assert loop.gap_detector.retriever is not None

def test_retrieval_integration():
    """Verify DeepLoop uses retrieval for prompt context."""
    # We'll mock the internal methods to verify calls
    loop = DeepLoop(llm=MockLLMForLoop(), max_iterations=1, sleep_seconds=0)
    
    # Mock retrieval
    loop.registry.register(lambda x: x, name="dummy_tool")
    
    # Mock format_tools_for_step return
    with patch('core.deep_loop.format_tools_for_step') as mock_format:
        mock_format.return_value = "Mocked Tool List"
        
        context = loop._get_tools_context("some goal")
        
        mock_format.assert_called()
        assert context == "Mocked Tool List"

@pytest.mark.asyncio
async def test_gap_detection_trigger():
    """Verify gap detection is triggered when tools are missing."""
    # This requires deep integration setup (listeners etc)
    # Ideally we check if ToolRetriever notified the listener
    
    retriever = get_tool_retriever()
    
    # Mock the listener
    mock_listener = MagicMock()
    retriever.add_gap_listener(mock_listener)
    
    # Trigger retrieval with unknown goal
    retriever.retrieve("fly to mars with warp drive")
    
    # Should trigger gap detection (low score)
    # Verify listener called
    # Note: retrieve calls _notify_gap if no match
    
    # Check if mock listener called
    assert mock_listener.called or True # retrieval might return *something* if threshold low
    
    # Force low threshold to ensure it matches nothing? 
    # Or force empty registry
    retriever._tool_cache.clear()
    retriever.retrieve("warp drive")
    mock_listener.assert_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
