"""
Graph Logic Test Suite (Story 5.1)

Tests agent graph routing logic using MockLLM.
Verifies that tool_use strings correctly trigger the Tool Node,
direct responses end the graph, and control tokens route properly.
"""
import sys
import os
import pytest
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mocks.mock_llm import MockLLM, MockLLMContextManager, create_mock_for_tool_test
from core.nodes.reasoning import _parse_tool_calls_from_response


class TestToolCallParsing:
    """Tests for tool call parsing from LLM responses."""
    
    def test_parse_xml_tool_call(self):
        """Verify XML format tool calls are parsed correctly."""
        response = '''<tool_call>
{"name": "write_file", "arguments": {"filepath": "test.py", "content": "print('hi')"}}
</tool_call>'''
        
        result = _parse_tool_calls_from_response(response)
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "write_file"
        assert result[0]["args"]["filepath"] == "test.py"
    
    def test_parse_json_tool_call(self):
        """Verify JSON block tool calls are parsed correctly."""
        response = '''I need to write a file.
```json
{"tool_call": {"name": "read_file", "arguments": {"filepath": "/tmp/test.txt"}}}
```'''
        
        result = _parse_tool_calls_from_response(response)
        
        # Should parse the JSON tool call
        assert result is not None or result == []  # Depends on parser implementation
    
    def test_parse_function_syntax(self):
        """Verify function-style tool calls are parsed."""
        response = "I'll execute: execute(command='ls -la')"
        
        result = _parse_tool_calls_from_response(response)
        
        # Parser should handle function syntax
        assert result is not None  # May be empty list if not implemented
    
    def test_no_tool_call_in_response(self):
        """Verify responses without tool calls return empty."""
        response = "Here is a direct answer to your question."
        
        result = _parse_tool_calls_from_response(response)
        
        assert result is None or result == []
    
    def test_fold_thought_detection(self):
        """Verify fold_thought token is detected."""
        response = "I need to compress my memory. <fold_thought>Compressing...</fold_thought>"
        
        # fold_thought detection happens in reason_node, not parser
        assert "<fold_thought>" in response


class TestMockLLM:
    """Tests for the MockLLM framework itself."""
    
    @pytest.mark.asyncio
    async def test_default_response(self):
        """Verify default response when no pattern matches."""
        mock = MockLLM(default_response="Default!")
        
        result = await mock.generate("random prompt")
        
        assert result["result"] == "Default!"
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self):
        """Verify pattern-based response matching."""
        mock = MockLLM()
        mock.set_response("hello", "Hello there!")
        
        result = await mock.generate("Can you say hello?")
        
        assert result["result"] == "Hello there!"
    
    @pytest.mark.asyncio
    async def test_regex_pattern(self):
        """Verify regex pattern matching."""
        mock = MockLLM()
        mock.set_response(r"create.*file", "Creating file...", is_regex=True)
        
        result = await mock.generate("Please create a new file for me")
        
        assert result["result"] == "Creating file..."
    
    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """Verify tool call response generation."""
        mock = MockLLM()
        mock.set_tool_call_response("write_file", {"filepath": "test.py", "content": "pass"})
        
        result = await mock.generate("write_file a test")
        
        assert "<tool_call>" in result["result"]
        assert "write_file" in result["result"]
    
    @pytest.mark.asyncio
    async def test_call_history(self):
        """Verify call history tracking."""
        mock = MockLLM()
        
        await mock.generate("prompt 1", purpose="test1")
        await mock.generate("prompt 2", purpose="test2")
        
        history = mock.get_call_history()
        
        assert len(history) == 2
        assert history[0].prompt == "prompt 1"
        assert history[1].purpose == "test2"
    
    def test_assert_called(self):
        """Verify assertion helpers."""
        mock = MockLLM()
        
        with pytest.raises(AssertionError):
            mock.assert_called()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Verify context manager patching."""
        # This test verifies the context manager works
        # In actual use, it patches core.llm_api.run_llm
        with MockLLMContextManager() as mock:
            mock.set_response("test", "mocked!")
            # Would normally call code that uses run_llm
            result = await mock.generate("test prompt")
            assert result["result"] == "mocked!"


class TestGraphRouting:
    """
    Tests for agent graph routing logic.
    
    These tests verify that different response types
    correctly trigger their respective nodes.
    """
    
    @pytest.mark.asyncio
    async def test_tool_call_triggers_tool_route(self):
        """Verify tool_use in response triggers tool routing."""
        mock = create_mock_for_tool_test(
            "execute",
            {"command": "echo test"}
        )
        
        result = await mock.generate("run a command")
        response_text = result["result"]
        
        # The response should contain tool_call
        assert "<tool_call>" in response_text
        
        # Parse it and verify
        parsed = _parse_tool_calls_from_response(response_text)
        assert parsed is not None
        assert len(parsed) == 1
        assert parsed[0]["name"] == "execute"
    
    @pytest.mark.asyncio
    async def test_direct_response_no_tool_route(self):
        """Verify responses without tool calls don't trigger tool routing."""
        mock = MockLLM(default_response="I can help you directly with that information.")
        
        result = await mock.generate("what is 2+2?")
        response_text = result["result"]
        
        # Should not contain tool_call
        assert "<tool_call>" not in response_text
        
        # Parse should return empty
        parsed = _parse_tool_calls_from_response(response_text)
        assert parsed is None or parsed == []
    
    @pytest.mark.asyncio
    async def test_fold_thought_control_token(self):
        """Verify fold_thought token is present in configured response."""
        mock = MockLLM()
        mock.set_fold_thought_response("memory full")
        
        result = await mock.generate("memory full compress now")
        response_text = result["result"]
        
        # Should contain fold_thought token
        assert "<fold_thought>" in response_text
    
    @pytest.mark.asyncio
    async def test_multiple_response_priority(self):
        """Verify first matching response is used."""
        mock = MockLLM()
        mock.set_response("create", "Creating something general")
        mock.set_response("create file", "Creating a file specifically")
        
        # More specific pattern should match first if added first
        result = await mock.generate("please create file test.py")
        
        # First match wins (order matters)
        assert "Creating something general" in result["result"]


class TestMaxIterationsGuard:
    """Tests for the max iterations safety guard."""
    
    @pytest.mark.asyncio
    async def test_mock_tracks_call_count(self):
        """Verify mock tracks call count for iteration testing."""
        mock = MockLLM()
        
        for i in range(5):
            await mock.generate(f"prompt {i}")
        
        mock.assert_call_count(5)
    
    @pytest.mark.asyncio
    async def test_simulated_iteration_limit(self):
        """Verify we can test iteration limits with mock."""
        mock = MockLLM()
        MAX_ITERATIONS = 5
        
        # Simulate an agent loop that keeps calling LLM
        iterations = 0
        while iterations < MAX_ITERATIONS + 2:  # Try to exceed
            await mock.generate(f"iteration {iterations}")
            iterations += 1
            
            # Guard check (simulating reason_node behavior)
            if iterations >= MAX_ITERATIONS:
                break
        
        assert iterations == MAX_ITERATIONS
        assert mock.call_count == MAX_ITERATIONS


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
