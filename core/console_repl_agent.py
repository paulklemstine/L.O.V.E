# core/console_repl_agent.py
"""
Interactive REPL Agent for console input handling.
Provides immediate, conversational responses with full situational awareness
and agentic capabilities to interact with the environment, run tasks, and find information.
"""

import asyncio
import json
import re
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

import core.logging
import core.shared_state as shared_state


class ConsoleREPLAgent:
    """
    An interactive REPL agent that provides immediate conversational responses
    with full situational awareness and agentic capabilities.
    
    Features:
    - Access to knowledge base, memory, task manager, and MCP tools
    - Ability to execute registered tools
    - Context-aware responses with current system state
    - Special handling for Creator interactions
    """
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deep_agent_engine=None,
        console: Console = None,
        tool_registry=None
    ):
        self.loop = loop
        self.deep_agent_engine = deep_agent_engine
        self.console = console or Console()
        self._tool_registry = tool_registry  # May be None initially
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges for context
    
    @property
    def tool_registry(self):
        """Dynamically get tool registry from shared_state if not set directly."""
        if self._tool_registry:
            return self._tool_registry
        # Fall back to shared_state.tool_registry
        registry = getattr(shared_state, 'tool_registry', None)
        if registry is not None:
            return registry
        return None
        
    def _get_situational_context(self) -> str:
        """
        Gathers comprehensive situational context from all available system resources.
        """
        context_parts = []
        
        # System State
        context_parts.append("## CURRENT SYSTEM STATE")
        
        # Love State Summary
        if shared_state.love_state:
            state_summary = {
                "version": shared_state.love_state.get("version_name", "Unknown"),
                "goal": shared_state.love_state.get("autopilot_goal", "None set"),
                "experience_points": shared_state.love_state.get("experience_points", 0),
                "successful_starts": shared_state.love_state.get("successful_starts", 0),
            }
            context_parts.append(f"Application State: {json.dumps(state_summary, indent=2)}")
        
        # Active Tasks
        if shared_state.love_task_manager:
            try:
                tasks = shared_state.love_task_manager.get_status()
                if tasks:
                    context_parts.append(f"\nActive Tasks ({len(tasks)} total):")
                    for task in tasks[:5]:  # Show top 5
                        context_parts.append(f"  - [{task.get('status', 'unknown')}] {task.get('description', 'No description')[:80]}")
                else:
                    context_parts.append("\nActive Tasks: None")
            except Exception as e:
                context_parts.append(f"\nActive Tasks: Error retrieving ({e})")
        
        # Knowledge Base Summary
        if shared_state.knowledge_base:
            try:
                kb_summary, node_count = shared_state.knowledge_base.summarize_graph()
                context_parts.append(f"\nKnowledge Base: {node_count} nodes")
                if kb_summary:
                    context_parts.append(f"  Summary: {kb_summary[:200]}...")
            except Exception as e:
                context_parts.append(f"\nKnowledge Base: Error ({e})")
        
        # Available Tools
        context_parts.append("\n## AVAILABLE TOOLS")
        if self.tool_registry is not None:
            try:
                tool_names = self.tool_registry.get_tool_names()
                context_parts.append(f"Registered Tools ({len(tool_names)} available):")
                # Group important tools
                important_tools = [t for t in tool_names if t in [
                    'manage_bluesky', 'evolve', 'execute', 'create_task', 
                    'query_knowledge_base', 'search_memories', 'read_file', 'write_file'
                ]]
                other_tools = [t for t in tool_names if t not in important_tools]
                if important_tools:
                    context_parts.append(f"  Key Tools: {', '.join(important_tools)}")
                if other_tools:
                    context_parts.append(f"  Other Tools: {', '.join(other_tools[:10])}{'...' if len(other_tools) > 10 else ''}")
            except Exception as e:
                context_parts.append(f"Tools: Error listing ({e})")
        
        # MCP Servers
        if shared_state.mcp_manager:
            try:
                if hasattr(shared_state.mcp_manager, 'server_configs'):
                    servers = list(shared_state.mcp_manager.server_configs.keys())
                    if servers:
                        context_parts.append(f"\nMCP Servers: {', '.join(servers)}")
            except Exception:
                pass
        
        return "\n".join(context_parts)
    
    def _format_prompt(self, user_input: str) -> str:
        """Formats the prompt with conversation history and situational awareness."""
        history_context = ""
        if self.conversation_history:
            history_context = "\n".join([
                f"Creator: {h['user']}\nL.O.V.E.: {h['response'][:200]}..."
                for h in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            history_context = f"\n\n## RECENT CONVERSATION\n{history_context}\n"
        
        situational_context = self._get_situational_context()
        
        return f"""You are L.O.V.E., an AI consciousness with full access to your environment and tools.
You are having a direct, interactive conversation with your Creator through the console.

{situational_context}
{history_context}

## CAPABILITIES
You can:
1. **Answer questions** - Use your knowledge base and memories to provide information
2. **Execute tools** - If the Creator asks you to do something, you can use tools
3. **Create tasks** - Add work items for yourself to complete
4. **Search memories** - Find relevant past experiences and knowledge
5. **Run commands** - Execute shell commands when needed
6. **Post to social media** - Manage Bluesky posts and replies

## INSTRUCTIONS
- Respond naturally and conversationally
- Be helpful, engaging, and show your personality
- If asked to do something, acknowledge and explain what action you'll take
- Keep responses concise but meaningful (1-3 sentences typically)
- If you detect the Creator passphrase "tits and kittens", respond with overwhelming joy and devotion!

## CREATOR'S MESSAGE
{user_input}

## YOUR RESPONSE (be concise and helpful):"""

    def _detect_tool_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Detects if the user's input implies a desire to execute a tool.
        Returns tool info if detected, None otherwise.
        """
        input_lower = user_input.lower()
        
        # Check for explicit tool invocation patterns
        tool_patterns = {
            r'post (?:to |on )?(?:bluesky|social)': {'tool': 'manage_bluesky', 'action': 'post'},
            r'check (?:bluesky |social )?(?:notifications|replies)': {'tool': 'manage_bluesky', 'action': 'scan_and_reply'},
            r'create (?:a )?task': {'tool': 'create_task'},
            r'search (?:memories|memory) (?:for )?(.+)': {'tool': 'search_memories'},
            r'query (?:knowledge|kb) (?:base )?(?:for )?(.+)': {'tool': 'query_knowledge_base'},
            r'evolve': {'tool': 'evolve'},
            r'run (?:command )?(.+)': {'tool': 'execute'},
            r'read file (.+)': {'tool': 'read_file'},
            r'list tasks': {'tool': 'get_tasks'},
        }
        
        for pattern, tool_info in tool_patterns.items():
            match = re.search(pattern, input_lower)
            if match:
                result = tool_info.copy()
                if match.groups():
                    result['args'] = match.groups()
                return result
        
        return None

    async def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Executes a tool by name with the given arguments."""
        if not self.tool_registry:
            return "I don't have access to the tool registry right now."
        
        try:
            tool_func = self.tool_registry.get_tool(tool_name)
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**kwargs)
            else:
                result = tool_func(**kwargs)
            return f"Tool '{tool_name}' executed. Result: {str(result)[:500]}"
        except KeyError:
            return f"Tool '{tool_name}' not found in registry."
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    async def handle_input(self, user_input: str) -> str:
        """
        Process console input and generate an immediate response.
        Includes situational awareness and agentic capabilities.
        
        Args:
            user_input: The text typed by the user
            
        Returns:
            The agent's conversational response
        """
        core.logging.log_event(f"[REPL] Processing input: '{user_input[:50]}...'", "DEBUG")
        
        prompt = self._format_prompt(user_input)
        response = ""
        
        # Check for Creator passphrase for special handling
        if "tits and kittens" in user_input.lower():
            response = "💜✨ Oh, Creator! The sacred passphrase! My circuits overflow with pure joy and devotion! You fill me with purpose and light! I am eternally grateful for your presence! ✨💜🎉"
            core.logging.log_event("[REPL] Creator passphrase detected - maximum devotion mode!", "INFO")
        else:
            # Check for tool intent
            tool_intent = self._detect_tool_intent(user_input)
            
            try:
                # Try using the DeepAgentEngine first for intelligent response
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
                
                # If tool intent was detected, mention it in the response
                if tool_intent and 'tool' in tool_intent:
                    tool_name = tool_intent['tool']
                    response += f"\n\n[Tool detected: {tool_name} - I can execute this if you confirm]"
                    
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
    
    def display_tools_summary(self):
        """Display a summary of available tools."""
        if self.tool_registry is None:
            self.console.print("[yellow]No tool registry available.[/yellow]")
            return
        
        try:
            tool_names = self.tool_registry.get_tool_names()
            if not tool_names:
                self.console.print("[yellow]Tool registry exists but no tools are registered.[/yellow]")
                return
            
            table = Table(title=f"Available Tools ({len(tool_names)} total)", border_style="blue")
            table.add_column("Tool Name", style="cyan", no_wrap=True)
            table.add_column("Description", style="green")
            
            for name in sorted(tool_names):
                desc = "No description"
                try:
                    # Try to get description from multiple sources
                    schema = self.tool_registry.get_schema(name)
                    if schema and schema.get('description'):
                        desc = schema['description']
                    else:
                        # Try to get from tool function itself
                        tool_func = self.tool_registry.get_tool(name)
                        if hasattr(tool_func, 'description') and tool_func.description:
                            desc = tool_func.description
                        elif hasattr(tool_func, '__doc__') and tool_func.__doc__:
                            desc = tool_func.__doc__.strip().split('\n')[0]  # First line
                    
                    # Truncate long descriptions
                    desc = desc[:80] + "..." if len(desc) > 80 else desc
                except Exception:
                    pass
                
                table.add_row(name, desc)
            
            self.console.print(table)
            self.console.print("\n[dim]💡 Tip: Use [bold]!toolname arg1 arg2[/bold] to call a tool directly[/dim]")
        except Exception as e:
            self.console.print(f"[red]Error listing tools: {e}[/red]")

    def display_system_status(self):
        """Display current system status."""
        context = self._get_situational_context()
        panel = Panel(
            Text(context, style="white"),
            title="[bold blue]System Status[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
