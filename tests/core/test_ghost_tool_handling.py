"""
Tests for Ghost Tool Handling and Duplicate Call Detection (Story 1.1).
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from core.nodes.execution import tool_execution_node
from core.state import DeepAgentState

class TestGhostToolHandling:
    
    @pytest.mark.asyncio
    async def test_duplicate_tool_call_prevention(self):
        """Verify that calling the same tool with exact same args is blocked."""
        # Setup state with a pre-executed call
        state = DeepAgentState(
            messages=[
                AIMessage(
                    content="", 
                    tool_calls=[{
                        "name": "test_tool", 
                        "args": {"x": 1}, 
                        "id": "call_2"
                    }]
                )
            ],
            executed_tool_calls=['test_tool:{"x": 1}'],
            loop_count=0
        )
        
        # Capture logging to verify warning
        with patch("core.logging.log_event") as mock_log:
            result = await tool_execution_node(state)
            
            # Check results
            outputs = result["messages"]
            assert len(outputs) == 1
            response = outputs[0].content
            
            # Should have an error about duplication
            assert "Error: You have already executed the tool" in response
            assert "test_tool" in response
            
            # Verify no new execution added (it was already there)
            # Actually, our logic appends only if not skipped, so it shouldn't dup in list?
            # Wait, `executed_tool_calls` in result matches inputs?
            # The list in state is mutated or copied? 
            # In our implementation: `executed_history = state.get("executed_tool_calls", [])`
            # `executed_set = set(executed_history)`
            # If dup, we DON'T add to set/list locally.
            # So returned executed_tool_calls should be same length as input if dup found.
            
            assert len(result["executed_tool_calls"]) == 1
            
    
    @pytest.mark.asyncio
    async def test_ghost_tool_educational_feedback(self):
        """Verify that calling a non-existent tool gives educational feedback."""
        state = DeepAgentState(
            messages=[
                AIMessage(
                    content="", 
                    tool_calls=[{
                        "name": "phantom_tool_9000", 
                        "args": {}, 
                        "id": "call_ghost"
                    }]
                )
            ],
            executed_tool_calls=[],
            loop_count=0
        )
        
        with patch("core.nodes.execution._get_tool_from_registry", return_value=None):
            result = await tool_execution_node(state)
            
            outputs = result["messages"]
            assert len(outputs) == 1
            response = outputs[0].content
            
            # Verify educational message
            assert "Tool 'phantom_tool_9000' not found" in response
            assert "Please use the 'retrieve_tools' tool" in response
            assert "Do not hallucinate tools" in response

    @pytest.mark.asyncio
    async def test_valid_execution_updates_history(self):
        """Verify valid execution adds to history."""
        state = DeepAgentState(
            messages=[
                AIMessage(
                    content="", 
                    tool_calls=[{
                        "name": "valid_tool", 
                        "args": {"a": "b"}, 
                        "id": "call_valid"
                    }]
                )
            ],
            executed_tool_calls=[],
            loop_count=0
        )
        
        mock_tool_func = MagicMock(return_value="Success")
        
        with patch("core.nodes.execution._get_tool_from_registry", return_value=mock_tool_func), \
             patch("core.nodes.execution._safe_execute_tool", return_value="Success"):
            
            result = await tool_execution_node(state)
            
            # Verify history updated
            history = result["executed_tool_calls"]
            assert len(history) == 1
            assert "valid_tool" in history[0]
            assert '{"a": "b"}' in history[0]
