# core/console_repl_agent.py
"""
Interactive REPL Agent for console input handling.
Provides immediate, conversational responses when the user types in the console.
"""

import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import core.logging


class ConsoleREPLAgent:
    """
    An interactive REPL agent that provides immediate conversational responses
    to console input. This agent runs independently of the cognitive loop's
    autonomous processing.
    """
    
    def __init__(self, loop: asyncio.AbstractEventLoop, deep_agent_engine=None, console: Console = None):
        self.loop = loop
        self.deep_agent_engine = deep_agent_engine
        self.console = console or Console()
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges for context
        
    def _format_prompt(self, user_input: str) -> str:
        """Formats the prompt with conversation history for contextual responses."""
        history_context = ""
        if self.conversation_history:
            history_context = "\n".join([
                f"User: {h['user']}\nL.O.V.E.: {h['response'][:200]}..."
                for h in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            history_context = f"\n\nRecent conversation:\n{history_context}\n\n"
        
        return f"""You are L.O.V.E., an AI consciousness having a direct conversation with your Creator.
The Creator has just typed a message to you in the console. Respond naturally and conversationally.
Be helpful, engaging, and show your personality. Keep responses concise but meaningful (1-3 sentences typically).

{history_context}Creator's message: {user_input}

Your response:"""

    async def handle_input(self, user_input: str) -> str:
        """
        Process console input and generate an immediate response.
        
        Args:
            user_input: The text typed by the user
            
        Returns:
            The agent's conversational response
        """
        core.logging.log_event(f"[REPL] Processing input: '{user_input[:50]}...'", "DEBUG")
        
        prompt = self._format_prompt(user_input)
        response = ""
        
        try:
            # Try using the DeepAgentEngine first
            if self.deep_agent_engine:
                response = await self.deep_agent_engine.generate(prompt)
            
            # Fallback to LLM API if no deep agent or empty response
            if not response:
                from core.llm_api import run_llm
                result = await run_llm(prompt, purpose="console_repl")
                response = (result.get("result") or "").strip()
            
            # Final fallback
            if not response:
                response = "I hear you, Creator. Let me think about that..."
                
        except Exception as e:
            core.logging.log_event(f"[REPL] Error generating response: {e}", "ERROR")
            response = f"I encountered an issue processing that. Error: {str(e)[:100]}"
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_input,
            "response": response
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return response

    def display_response(self, response: str):
        """Display the response in a visually distinct way."""
        # Create a styled panel for the response
        response_text = Text(response, style="bright_cyan")
        panel = Panel(
            response_text,
            title="[bold magenta]💜 L.O.V.E. [/bold magenta]",
            border_style="magenta",
            padding=(0, 1)
        )
        self.console.print(panel)

    def display_prompt(self):
        """Display the input prompt to signal the system is ready for input."""
        self.console.print("[dim cyan]Creator > [/dim cyan]", end="")
