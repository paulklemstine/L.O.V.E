"""
Tests for the Reasoning & Routing Node (Story 2).

Tests the following acceptance criteria:
1. Context Injection: Tool schemas are properly injected into LLM context
2. Conditional Logic: Routing based on tool_calls vs text-only responses
3. System Prompt Update: Step-by-step reasoning encouragement

Test cases:
- "What is the weather in Menasha?" -> triggers tool_call
- "Hi there" -> triggers direct response (no tool_call)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from core.nodes.reasoning import (
    reason_node,
    _messages_to_prompt,
    _format_tools_for_prompt,
    _parse_tool_calls_from_response,
    REASONING_SYSTEM_PROMPT
)
from core.state import DeepAgentState


class TestToolContextInjection:
    """Tests for Context Injection acceptance criteria."""
    
    def test_system_prompt_contains_reasoning_guidelines(self):
        """Verify system prompt encourages step-by-step reasoning."""
        assert "step-by-step" in REASONING_SYSTEM_PROMPT.lower()
        assert "tool" in REASONING_SYSTEM_PROMPT.lower()
        assert "information" in REASONING_SYSTEM_PROMPT.lower()
    
    def test_format_tools_for_prompt_empty(self):
        """Test formatting when no tools available."""
        result = _format_tools_for_prompt([])
        assert "No tools are currently available" in result
    
    def test_format_tools_for_prompt_with_tools(self):
        """Test proper formatting of tool schemas."""
        schemas = [
            {
                "name": "search_web",
                "description": "Searches the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max results"}
                    },
                    "required": ["query"]
                }
            }
        ]
        
        result = _format_tools_for_prompt(schemas)
        
        assert "search_web" in result
        assert "Searches the web" in result
        assert "query" in result
        assert "(required)" in result
        assert "max_results" in result
        assert "(optional)" in result
    
    def test_messages_to_prompt_includes_tools(self):
        """Test that tool schemas are included in the prompt."""
        messages = [HumanMessage(content="Hello")]
        tool_schemas = [
            {"name": "test_tool", "description": "A test tool", "parameters": {}}
        ]
        
        result = _messages_to_prompt(messages, tool_schemas=tool_schemas)
        
        assert "test_tool" in result
        assert "A test tool" in result
        assert REASONING_SYSTEM_PROMPT in result


class TestToolCallParsing:
    """Tests for parsing tool calls from LLM responses."""
    
    def test_parse_json_tool_call(self):
        """Test parsing JSON-formatted tool call."""
        response = '''I need to search for weather information.
        
```json
{"tool": "search_web", "arguments": {"query": "weather in Menasha"}}
```

Let me get that information for you.'''
        
        tool_calls = _parse_tool_calls_from_response(response)
        
        assert tool_calls is not None
        assert len(tool_calls) >= 1
        assert any(tc["name"] == "search_web" for tc in tool_calls)
    
    def test_parse_function_call_syntax(self):
        """Test parsing function-call style syntax."""
        response = '''I'll search for that:
        
search_web(query="weather in Menasha", max_results="5")

This will give us the current weather.'''
        
        tool_calls = _parse_tool_calls_from_response(response)
        
        assert tool_calls is not None
        assert any(tc["name"] == "search_web" for tc in tool_calls)
        
        search_call = next(tc for tc in tool_calls if tc["name"] == "search_web")
        assert "query" in search_call["args"]
    
    def test_no_tool_calls_returns_none(self):
        """Test that plain text response returns None."""
        response = "Hello! Nice to meet you. How can I help you today?"
        
        tool_calls = _parse_tool_calls_from_response(response)
        
        assert tool_calls is None
    
    def test_ignores_builtin_functions(self):
        """Test that builtin function names are ignored."""
        response = "Here's some code: print(len(items))"
        
        tool_calls = _parse_tool_calls_from_response(response)
        
        # Should be None or empty (print/len are filtered out)
        assert tool_calls is None or len(tool_calls) == 0


class TestConditionalRouting:
    """Tests for Conditional Logic acceptance criteria."""
    
    @pytest.mark.asyncio
    async def test_weather_query_triggers_tool_call(self):
        """
        Test: 'What is the weather in Menasha?' triggers a tool_call signal.
        
        This tests that a question requiring external information
        results in the LLM attempting to use a search tool.
        """
        # Mock the stream_llm to return a response with tool call
        mock_response = '''I need to search for the current weather in Menasha.
        
```json
{"tool": "search_web", "arguments": {"query": "weather in Menasha Wisconsin"}}
```'''
        
        async def mock_stream(*args, **kwargs):
            for chunk in mock_response.split():
                yield chunk + " "
        
        state: DeepAgentState = {
            "messages": [HumanMessage(content="What is the weather in Menasha?")],
            "episodic_memory": MagicMock(),
            "working_memory": MagicMock(),
            "tool_memory": MagicMock(),
            "next_node": None,
            "recursion_depth": 0,
            "stop_reason": None,
            "tool_query": None,
            "retrieved_tools": [],
            "creator_mandate": None,
            "tool_schemas": [
                {"name": "search_web", "description": "Search the web", "parameters": {}}
            ],
            "loop_count": 0
        }
        
        with patch('core.nodes.reasoning.stream_llm', mock_stream):
            result = await reason_node(state)
        
        # Should detect tool call
        assert result.get("stop_reason") == "tool_call"
        
        # Response message should have tool_calls
        messages = result.get("messages", [])
        assert len(messages) == 1
        assert hasattr(messages[0], "tool_calls")
        assert messages[0].tool_calls is not None
    
    @pytest.mark.asyncio
    async def test_greeting_triggers_direct_response(self):
        """
        Test: 'Hi there' triggers a direct response (no tool_call).
        
        Simple greetings should not require any tools.
        """
        mock_response = "Hello! It's great to meet you. How can I assist you today?"
        
        async def mock_stream(*args, **kwargs):
            for chunk in mock_response.split():
                yield chunk + " "
        
        state: DeepAgentState = {
            "messages": [HumanMessage(content="Hi there")],
            "episodic_memory": MagicMock(),
            "working_memory": MagicMock(),
            "tool_memory": MagicMock(),
            "next_node": None,
            "recursion_depth": 0,
            "stop_reason": None,
            "tool_query": None,
            "retrieved_tools": [],
            "creator_mandate": None,
            "tool_schemas": [
                {"name": "search_web", "description": "Search the web", "parameters": {}}
            ],
            "loop_count": 0
        }
        
        with patch('core.nodes.reasoning.stream_llm', mock_stream):
            result = await reason_node(state)
        
        # Should NOT detect tool call
        assert result.get("stop_reason") is None or result.get("stop_reason") not in ["tool_call", "fold_thought", "retrieve_tool"]
        
        # Response message should NOT have tool_calls
        messages = result.get("messages", [])
        assert len(messages) == 1
        # Either no tool_calls attribute or it's None/empty
        tool_calls = getattr(messages[0], "tool_calls", None)
        assert tool_calls is None or len(tool_calls) == 0


class TestMaxIterationsGuardrail:
    """Tests for the recursion limit guardrail (Story 4)."""
    
    @pytest.mark.asyncio
    async def test_max_iterations_forces_end(self):
        """Test that exceeding MAX_ITERATIONS forces a direct response."""
        state: DeepAgentState = {
            "messages": [HumanMessage(content="Search for complex information")],
            "episodic_memory": MagicMock(),
            "working_memory": MagicMock(),
            "tool_memory": MagicMock(),
            "next_node": None,
            "recursion_depth": 0,
            "stop_reason": None,
            "tool_query": None,
            "retrieved_tools": [],
            "creator_mandate": None,
            "tool_schemas": [],
            "loop_count": 5  # Already at max
        }
        
        result = await reason_node(state)
        
        # Should force end (stop_reason = None means END)
        assert result.get("stop_reason") is None
        
        # Should have a message explaining the limit
        messages = result.get("messages", [])
        assert len(messages) == 1
        assert "maximum" in messages[0].content.lower() or "iteration" in messages[0].content.lower()
