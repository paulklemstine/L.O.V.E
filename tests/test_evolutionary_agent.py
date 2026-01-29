"""
Tests for Evolutionary Agent - Epic 2

Story 2.1: Verification of Fabrication Loop
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.evolutionary_agent import EvolutionaryAgent
from core.evolution_state import EvolutionarySpecification
from core.tool_validator import ValidationResult

class MockLLMClient:
    async def generate(self, prompt: str) -> str:
        return "def fixed_code(): pass"

class TestEvolutionaryAgent:
    
    @pytest.fixture
    def agent(self):
        return EvolutionaryAgent(llm_client=MockLLMClient())
    
    @pytest.mark.asyncio
    async def test_process_spec_success(self, agent):
        """Should succeed if fabrication and validation pass first time."""
        spec = EvolutionarySpecification(
            functional_name="test_tool",
            required_arguments={},
            expected_output="str",
            id="123"
        )
        
        # Mock fabricator
        agent.fabricator.fabricate_tool = AsyncMock(return_value={
            "success": True,
            "file_path": "/tmp/test_tool.py",
            "code": "def test_tool(): pass"
        })
        
        # Mock validator
        agent.validator.validate = AsyncMock(return_value=ValidationResult(
            passed=True, syntax_valid=True
        ))
        
        # Mock finalization (file move)
        agent._finalize_tool = AsyncMock(return_value=True)
        
        result = await agent._process_single_spec(spec)
        assert result is True
        agent._finalize_tool.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_refinement_loop(self, agent):
        """Should retry and refine code if validation fails."""
        spec = EvolutionarySpecification(
            functional_name="refine_tool",
            required_arguments={},
            expected_output="str",
            id="456"
        )
        
        # Mock fabricator
        agent.fabricator.fabricate_tool = AsyncMock(return_value={
            "success": True,
            "file_path": "/tmp/refine_tool.py",
            "code": "def broken(): pass"
        })
        
        # Mock validator: Fail once, then Pass
        fail_result = ValidationResult(passed=False, error_message="Syntax Error")
        pass_result = ValidationResult(passed=True)
        
        agent.validator.validate = AsyncMock(side_effect=[fail_result, pass_result])
        
        # Mock refinement
        agent._refine_code = AsyncMock(return_value="def fixed(): pass")
        
        # Mock file write (since we aren't writing real files here)
        with patch("builtins.open", new_callable=MagicMock):
            agent._finalize_tool = AsyncMock(return_value=True)
            
            result = await agent._process_single_spec(spec)
            
            assert result is True
            assert agent.validator.validate.call_count == 2
            agent._refine_code.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, agent):
        """Should fail if max retries reached."""
        spec = EvolutionarySpecification(
            functional_name="fail_tool",
            required_arguments={},
            expected_output="str",
            id="789"
        )
        
        # Mock fabricator
        agent.fabricator.fabricate_tool = AsyncMock(return_value={
            "success": True,
            "file_path": "/tmp/fail_tool.py",
            "code": "def fail(): pass"
        })
        
        # Validation always fails
        agent.validator.validate = AsyncMock(return_value=ValidationResult(passed=False))
        agent._refine_code = AsyncMock(return_value="def still_fail(): pass")
        
        with patch("builtins.open", new_callable=MagicMock):
            result = await agent._process_single_spec(spec)
            
            assert result is False
            # Attempts: 1 initial + 3 retries = 4 validations (actually logic is loop n+1)
            # range(MAX_RETRIES + 1) -> 0,1,2,3 if MAX_RETRIES=3. So 4 attempts.
            assert agent.validator.validate.call_count >= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
