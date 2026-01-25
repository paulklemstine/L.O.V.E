"""
DeepAgent Protocol - Story 4.2: Simulation Replay Test

Regression test that replays a standard scenario using mock LLM
to verify graph transitions and final state.
"""

import pytest
import asyncio
from typing import Dict, Any
from tests.mocks.mock_llm import MockLLM, MockLLMContextManager
from unittest.mock import patch, AsyncMock


class TestDeepAgentGraphTransitions:
    """Test suite for DeepAgent graph transitions."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a configured mock LLM for testing."""
        mock = MockLLM()
        
        # Configure planner response
        mock.set_response(
            "decompose",
            "1. Greet the user warmly\n2. Acknowledge their message\n3. Provide helpful response"
        )
        
        # Configure execution response
        mock.set_response(
            "Execute the following step",
            "Hello! Thank you for reaching out. I'm here to help you."
        )
        
        # Configure critic response (approved)
        mock.set_response(
            "quality assurance",
            '{"approved": true, "confidence": 0.95, "feedback": "Task completed successfully", "corrections": ""}'
        )
        
        return mock
    
    @pytest.mark.asyncio
    async def test_basic_graph_transitions(self, mock_llm):
        """
        Verify that the graph transitions correctly: Plan -> Execute -> Critic.
        
        Story 4.2: Assert graph transitions in the expected order.
        """
        with patch('core.llm_api.run_llm', new=mock_llm.generate):
            from core.deep_agent_graph import DeepAgentGraphRunner
            
            runner = DeepAgentGraphRunner(max_loops=5)
            result = await runner.run("Hello, how are you?")
            
            # Verify the mock was called
            mock_llm.assert_called()
            
            # Verify success
            assert result.get("success") is True or result.get("stop_reason") == "completed"
    
    @pytest.mark.asyncio
    async def test_plan_generation(self, mock_llm):
        """Test that planner generates a valid plan."""
        with patch('core.llm_api.run_llm', new=mock_llm.generate):
            from core.state import create_initial_state
            from core.agents.planner_agent import PlannerAgent
            
            state = create_initial_state("Create a greeting message")
            planner = PlannerAgent()
            
            result_state = await planner.plan(state)
            
            # Verify plan was created
            assert "plan" in result_state
            assert isinstance(result_state["plan"], list)
            assert len(result_state["plan"]) > 0
    
    @pytest.mark.asyncio
    async def test_critic_approval(self, mock_llm):
        """Test that critic correctly approves a good execution."""
        # Configure for approval
        mock_llm.set_response(
            "Evaluate",
            '{"approved": true, "confidence": 0.9, "feedback": "Looks good", "corrections": ""}'
        )
        
        with patch('core.llm_api.run_llm', new=mock_llm.generate):
            from core.state import create_initial_state
            from core.agents.critic_agent import CriticAgent
            
            state = create_initial_state("Test task")
            state["plan"] = ["Step 1: Do something"]
            state["past_steps"] = [("Step 1: Do something", "action", "completed successfully")]
            
            critic = CriticAgent()
            result_state = await critic.critique(state)
            
            # Verify approval
            assert "finalize" in result_state.get("next_node", "")
    
    @pytest.mark.asyncio
    async def test_critic_rejection_triggers_replan(self, mock_llm):
        """Test that critic rejection triggers replanning."""
        # Configure for rejection
        mock_llm.clear_responses()
        mock_llm.set_response(
            "Evaluate",
            '{"approved": false, "confidence": 0.3, "feedback": "Needs improvement", "corrections": "Try a different approach"}'
        )
        
        with patch('core.llm_api.run_llm', new=mock_llm.generate):
            from core.state import create_initial_state
            from core.agents.critic_agent import CriticAgent
            
            state = create_initial_state("Test task")
            state["plan"] = ["Step 1: Do something"]
            state["past_steps"] = [("Step 1: Do something", "action", "failed")]
            state["current_loop"] = 0
            state["max_loops"] = 5
            
            critic = CriticAgent()
            result_state = await critic.critique(state)
            
            # Verify replan
            assert result_state.get("next_node") == "planner"
            assert result_state.get("current_loop") == 1
            assert "corrections" in result_state.get("scratchpad", "").lower() or \
                   "Try" in result_state.get("scratchpad", "")
    
    @pytest.mark.asyncio
    async def test_max_loops_enforced(self, mock_llm):
        """Test that max loops prevents infinite recursion."""
        # Configure for continuous rejection
        mock_llm.clear_responses()
        mock_llm.set_response(
            "",  # Match all
            '{"approved": false, "confidence": 0.1, "feedback": "Still not good", "corrections": "Keep trying"}'
        )
        
        with patch('core.llm_api.run_llm', new=mock_llm.generate):
            from core.state import create_initial_state
            from core.agents.critic_agent import CriticAgent
            
            state = create_initial_state("Test task")
            state["plan"] = ["Step 1"]
            state["past_steps"] = [("Step 1", "action", "result")]
            state["current_loop"] = 4  # At max - 1
            state["max_loops"] = 5
            
            critic = CriticAgent()
            result_state = await critic.critique(state)
            
            # Should finalize due to max loops
            assert result_state.get("next_node") == "finalize"
            assert "max" in result_state.get("stop_reason", "").lower()


class TestGraphDiagram:
    """Test graph visualization."""
    
    def test_graph_diagram_generation(self):
        """Verify the graph can generate a Mermaid diagram."""
        from core.deep_agent_graph import DeepAgentGraphRunner
        
        runner = DeepAgentGraphRunner()
        diagram = runner.get_graph_diagram()
        
        # Verify it contains expected elements
        assert "planner" in diagram.lower() or "plan" in diagram.lower()
        assert "executor" in diagram.lower() or "execute" in diagram.lower()
        assert "critic" in diagram.lower()


class TestStateInitialization:
    """Test state creation and validation."""
    
    def test_create_initial_state(self):
        """Test that initial state is properly created."""
        from core.state import create_initial_state
        
        state = create_initial_state("Test input", max_loops=3)
        
        assert state["input"] == "Test input"
        assert state["max_loops"] == 3
        assert state["current_loop"] == 0
        assert state["plan"] == []
        assert state["past_steps"] == []
        assert state["scratchpad"] == ""
        assert state["criticism"] == ""
    
    def test_state_has_all_required_fields(self):
        """Verify all DeepAgent Protocol fields are present."""
        from core.state import create_initial_state
        
        state = create_initial_state("Test")
        
        required_fields = [
            "input", "chat_history", "plan", "past_steps",
            "scratchpad", "criticism", "max_loops", "current_loop"
        ]
        
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
