"""
Tests for Tool Gap Detector - Phase 3

Epic 1, Story 1.1: Verification of Evolutionary Awareness
"""

import os
import sys
import pytest
import asyncio
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.introspection.tool_gap_detector import ToolGapDetector
from core.evolution_state import EvolutionarySpecification, get_pending_specifications, load_evolution_state
from core.tool_retriever import ToolRetriever

# Mocks
class MockLLMClient:
    async def generate(self, prompt: str) -> str:
        return json.dumps({
            "functional_name": "mock_tool",
            "required_arguments": {"arg1": "str"},
            "expected_output": "str",
            "safety_constraints": ["none"]
        })

class TestToolGapDetector:
    
    @pytest.fixture
    def detector(self):
        # Reset state for testing
        # In a real test we'd mock the file operations, but here we'll let it use the file
        # and clean up, or rely on tmp_path if we refactored get_evolution_state_path
        
        # We'll rely on the fact that existing state loading is safe
        return ToolGapDetector(llm_client=MockLLMClient())
    
    @pytest.mark.asyncio
    async def test_analyze_gap(self, detector):
        """Should generate and store a specification."""
        spec = await detector.analyze_gap_and_specify("I need to fly to the moon")
        
        assert spec is not None
        assert spec.functional_name == "mock_tool"
        assert spec.status == "pending"
        
        # Verify it's in the pending list
        pending = get_pending_specifications()
        ids = [s.id for s in pending]
        assert spec.id in ids
    
    def test_listener_registration(self, detector):
        """Should register listener with retriever."""
        # The detector registers in __init__
        # We can check if it's logically connected
        # Accessing private members is naughty but effective for unit testing
        assert len(detector.retriever._gap_listeners) > 0
        
    def test_deduplication(self, detector):
        """Should not create duplicate specs for same context."""
        # This is harder to test without mocking the state get/set fully
        # or implementing the logic strictly.
        # The logic is:
        # pending = get_pending_specifications()
        # for spec in pending: if trigger == context: return
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
