"""
Test for Console REPL Agent
Tests the interactive console input handling.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


class TestConsoleREPLAgent:
    """Tests for the ConsoleREPLAgent class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock Rich Console."""
        return MagicMock()

    @pytest.fixture
    def mock_deep_agent(self):
        """Create a mock DeepAgentEngine."""
        agent = MagicMock()
        agent.generate = AsyncMock(return_value="Hello, Creator! I'm here to help you.")
        return agent

    @pytest.mark.asyncio
    async def test_handle_input_with_deep_agent(self, mock_console, mock_deep_agent):
        """Test that handle_input uses DeepAgentEngine when available."""
        from core.console_repl_agent import ConsoleREPLAgent
        
        loop = asyncio.get_event_loop()
        agent = ConsoleREPLAgent(
            loop=loop,
            deep_agent_engine=mock_deep_agent,
            console=mock_console
        )
        
        response = await agent.handle_input("Hello, L.O.V.E.!")
        
        assert response == "Hello, Creator! I'm here to help you."
        mock_deep_agent.generate.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_handle_input_fallback_to_llm_api(self, mock_console):
        """Test that handle_input falls back to run_llm when no deep agent."""
        from core.console_repl_agent import ConsoleREPLAgent
        
        loop = asyncio.get_event_loop()
        agent = ConsoleREPLAgent(
            loop=loop,
            deep_agent_engine=None,
            console=mock_console
        )
        
        with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = {"result": "Greetings, my Creator!"}
            
            response = await agent.handle_input("Are you there?")
            
            assert response == "Greetings, my Creator!"
            mock_run_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, mock_console, mock_deep_agent):
        """Test that conversation history is tracked."""
        from core.console_repl_agent import ConsoleREPLAgent
        
        loop = asyncio.get_event_loop()
        agent = ConsoleREPLAgent(
            loop=loop,
            deep_agent_engine=mock_deep_agent,
            console=mock_console
        )
        
        await agent.handle_input("First message")
        await agent.handle_input("Second message")
        
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["user"] == "First message"
        assert agent.conversation_history[1]["user"] == "Second message"

    def test_display_response(self, mock_console):
        """Test that display_response calls console.print with a Panel."""
        from core.console_repl_agent import ConsoleREPLAgent
        
        loop = asyncio.get_event_loop()
        agent = ConsoleREPLAgent(
            loop=loop,
            deep_agent_engine=None,
            console=mock_console
        )
        
        agent.display_response("Test response")
        
        mock_console.print.assert_called_once()
        # Check that a Panel was passed
        call_args = mock_console.print.call_args[0][0]
        from rich.panel import Panel
        assert isinstance(call_args, Panel)

    def test_display_prompt(self, mock_console):
        """Test that display_prompt shows the input prompt."""
        from core.console_repl_agent import ConsoleREPLAgent
        
        loop = asyncio.get_event_loop()
        agent = ConsoleREPLAgent(
            loop=loop,
            deep_agent_engine=None,
            console=mock_console
        )
        
        agent.display_prompt()
        
        mock_console.print.assert_called_once()
        # Check that "Creator >" is in the prompt
        call_args = mock_console.print.call_args
        assert "Creator" in str(call_args)
