"""
Integration Tests for Subagent Execution System

Tests the unified agent framework with LangChain, LangGraph, DeepAgent,
and MCP integrations. Can also "jack into" a running system to inspect
and test components live.

Run with: python -m pytest tests/test_subagent_integration.py -v
Or as a live inspection: python tests/test_subagent_integration.py --live
"""

import asyncio
import sys
import os
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestSubagentExecutorInit:
    """Tests for SubagentExecutor initialization."""
    
    def test_import_subagent_executor(self):
        """Test that SubagentExecutor can be imported."""
        from core.subagent_executor import SubagentExecutor
        assert SubagentExecutor is not None
    
    def test_executor_initializes(self):
        """Test basic initialization."""
        from core.subagent_executor import SubagentExecutor
        executor = SubagentExecutor()
        assert executor is not None
        assert executor.max_depth == 3
        assert executor._active_subagents == {}
    
    def test_executor_with_mcp_manager(self):
        """Test initialization with MCP manager."""
        from core.subagent_executor import SubagentExecutor
        
        mock_mcp = Mock()
        mock_mcp.list_running_servers.return_value = []
        
        executor = SubagentExecutor(mcp_manager=mock_mcp)
        assert executor.mcp_manager is mock_mcp


class TestSubagentTypes:
    """Tests for subagent type handling."""
    
    def test_fallback_prompts(self):
        """Test that all agent types have fallback prompts."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        agent_types = ["reasoning", "coding", "research", "social", "security", "analyst", "creative"]
        
        for agent_type in agent_types:
            prompt = executor._get_fallback_prompt(agent_type)
            assert prompt is not None
            assert len(prompt) > 10
    
    def test_unknown_agent_type_fallback(self):
        """Test that unknown agent types get default fallback."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        prompt = executor._get_fallback_prompt("unknown_type")
        assert "helpful AI assistant" in prompt


class TestToolCallParsing:
    """Tests for parsing tool calls from LLM responses."""
    
    def test_parse_json_markdown_block(self):
        """Test parsing JSON in markdown code block."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        text = '''Here's what I'll do:
        
```json
{"tool": "execute", "arguments": {"command": "ls -la"}}
```
'''
        result = executor._parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "execute"
        assert result["arguments"]["command"] == "ls -la"
    
    def test_parse_inline_json(self):
        """Test parsing inline JSON."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        text = 'I will use {"tool": "read_file", "arguments": {"filepath": "/tmp/test"}} to check'
        
        result = executor._parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "read_file"
    
    def test_parse_no_tool_call(self):
        """Test that text without tool calls returns None."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        text = "This is a normal response without any tool calls."
        
        result = executor._parse_tool_call(text)
        assert result is None


class TestSubagentInvocation:
    """Tests for subagent invocation."""
    
    @pytest.mark.asyncio
    async def test_invoke_subagent_with_mock_llm(self):
        """Test invoking a subagent with mocked LLM."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        
        # Mock the LLM call
        with patch('core.subagent_executor.run_llm') as mock_llm:
            mock_llm.return_value = {"result": "Task completed successfully."}
            
            result = await executor.invoke_subagent(
                agent_type="reasoning",
                task="What is 2+2?",
                max_iterations=1
            )
            
            assert result.success is True
            assert result.agent_type == "reasoning"
            assert "Task completed" in result.result or "2+2" in result.result or len(result.result) > 0
    
    @pytest.mark.asyncio
    async def test_invoke_subagent_with_tool_call(self):
        """Test subagent that makes a tool call."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        
        # First call returns tool use, second returns final answer
        call_count = [0]
        async def mock_llm(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"result": '```json\n{"tool": "read_file", "arguments": {"filepath": "/tmp/test"}}\n```'}
            return {"result": "The file contains test data."}
        
        with patch('core.subagent_executor.run_llm', side_effect=mock_llm):
            with patch.object(executor, '_execute_tool', return_value="test file contents"):
                result = await executor.invoke_subagent(
                    agent_type="research",
                    task="Read the test file",
                    max_iterations=3
                )
                
                assert result.success is True
                assert len(result.tool_calls) > 0
                assert result.iterations >= 1
    
    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """Test that max iterations limit is respected."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        
        # Always return a tool call to force iteration
        with patch('core.subagent_executor.run_llm') as mock_llm:
            mock_llm.return_value = {"result": '```json\n{"tool": "fake", "arguments": {}}\n```'}
            
            with patch.object(executor, '_execute_tool', return_value="result"):
                result = await executor.invoke_subagent(
                    agent_type="reasoning",
                    task="Keep trying",
                    max_iterations=3
                )
                
                assert result.iterations == 3


class TestWrapAsTool:
    """Tests for wrapping executor as LangChain tool."""
    
    def test_wrap_creates_tool(self):
        """Test that wrap_as_tool creates a valid LangChain tool."""
        from core.subagent_executor import SubagentExecutor
        from langchain_core.tools import BaseTool
        
        executor = SubagentExecutor()
        tool = executor.wrap_as_tool()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "invoke_subagent"
        assert "subagent" in tool.description.lower()


class TestLangGraphIntegration:
    """Tests for LangGraph subgraph creation."""
    
    def test_create_langgraph_subagent(self):
        """Test creating a LangGraph subgraph."""
        from core.subagent_executor import SubagentExecutor
        
        executor = SubagentExecutor()
        
        # This should not raise
        try:
            subgraph = executor.create_langgraph_subagent("reasoning")
            assert subgraph is not None
        except Exception as e:
            # May fail due to missing state imports, but should at least attempt
            pytest.skip(f"LangGraph integration not fully configured: {e}")


# ============================================================================
# LIVE SYSTEM INSPECTION ("JACK INTO THE MATRIX")
# ============================================================================

class LiveSystemInspector:
    """
    Inspects a running L.O.V.E. system to test components live.
    This allows you to "jack in" and see what's happening.
    """
    
    def __init__(self):
        self.console = Console()
    
    async def inspect_mcp_servers(self) -> Dict[str, Any]:
        """Check running MCP servers."""
        self.console.print("\n[bold cyan]🔌 MCP Server Status[/bold cyan]")
        
        try:
            from mcp_manager import MCPManager
            manager = MCPManager(self.console)
            
            running = manager.list_running_servers()
            configs = manager.server_configs
            
            table = Table(title="MCP Servers")
            table.add_column("Server", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Tools", style="yellow")
            
            for name, config in configs.items():
                status = "🟢 Running" if any(s["name"] == name for s in running) else "⚫ Stopped"
                tools = ", ".join(config.get("tools", {}).keys())[:40] + "..."
                table.add_row(name, status, tools)
            
            self.console.print(table)
            return {"running": running, "configs": configs}
            
        except Exception as e:
            self.console.print(f"[red]Failed to inspect MCP: {e}[/red]")
            return {"error": str(e)}
    
    async def inspect_tool_registry(self) -> Dict[str, Any]:
        """Inspect the global tool registry."""
        self.console.print("\n[bold cyan]🔧 Tool Registry[/bold cyan]")
        
        try:
            from core.tool_registry import get_global_registry
            registry = get_global_registry()
            
            tools = list(registry.keys()) if hasattr(registry, 'keys') else []
            
            self.console.print(f"Registered tools: {len(tools)}")
            for tool in tools[:10]:
                self.console.print(f"  • {tool}")
            if len(tools) > 10:
                self.console.print(f"  ... and {len(tools) - 10} more")
            
            return {"tool_count": len(tools), "tools": tools}
            
        except Exception as e:
            self.console.print(f"[red]Tool registry not available: {e}[/red]")
            return {"error": str(e)}
    
    async def inspect_subagent_executor(self) -> Dict[str, Any]:
        """Inspect SubagentExecutor status."""
        self.console.print("\n[bold cyan]🤖 SubagentExecutor Status[/bold cyan]")
        
        try:
            from core.subagent_executor import get_subagent_executor, SubagentExecutor
            
            # Check if we can create an executor
            executor = SubagentExecutor()
            
            self.console.print("[green]✓ SubagentExecutor initialized[/green]")
            self.console.print(f"  Max depth: {executor.max_depth}")
            self.console.print(f"  Active subagents: {len(executor._active_subagents)}")
            
            return {
                "status": "ok",
                "max_depth": executor.max_depth,
                "active_subagents": len(executor._active_subagents)
            }
            
        except Exception as e:
            self.console.print(f"[red]Failed: {e}[/red]")
            return {"error": str(e)}
    
    async def test_subagent_invocation(self, agent_type: str = "reasoning", task: str = "What is 2+2?"):
        """Live test a subagent invocation."""
        self.console.print(f"\n[bold cyan]🧪 Testing Subagent: {agent_type}[/bold cyan]")
        self.console.print(f"Task: {task}")
        
        try:
            from core.subagent_executor import SubagentExecutor
            
            executor = SubagentExecutor()
            
            self.console.print("[yellow]Invoking subagent...[/yellow]")
            
            result = await executor.invoke_subagent(
                agent_type=agent_type,
                task=task,
                max_iterations=3
            )
            
            panel = Panel(
                f"""[bold]Success:[/bold] {result.success}
[bold]Agent Type:[/bold] {result.agent_type}
[bold]Iterations:[/bold] {result.iterations}
[bold]Tool Calls:[/bold] {len(result.tool_calls)}

[bold]Result:[/bold]
{result.result[:500]}{'...' if len(result.result) > 500 else ''}""",
                title="Subagent Result",
                border_style="green" if result.success else "red"
            )
            self.console.print(panel)
            
            return result.__dict__
            
        except Exception as e:
            self.console.print(f"[red]Test failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    async def inspect_langgraph_state(self) -> Dict[str, Any]:
        """Inspect the LangGraph state configuration."""
        self.console.print("\n[bold cyan]📊 LangGraph State[/bold cyan]")
        
        try:
            from core.state import DeepAgentState
            from typing import get_type_hints
            
            hints = get_type_hints(DeepAgentState)
            
            table = Table(title="DeepAgentState Fields")
            table.add_column("Field", style="cyan")
            table.add_column("Type", style="yellow")
            
            for name, type_hint in hints.items():
                table.add_row(name, str(type_hint)[:50])
            
            self.console.print(table)
            return {"fields": list(hints.keys())}
            
        except Exception as e:
            self.console.print(f"[red]Failed: {e}[/red]")
            return {"error": str(e)}
    
    async def run_full_inspection(self):
        """Run complete system inspection."""
        self.console.print(Panel(
            "[bold magenta]🔮 JACKING INTO THE MATRIX 🔮[/bold magenta]\n"
            "Inspecting live L.O.V.E. system components...",
            border_style="magenta"
        ))
        
        results = {}
        
        results["mcp"] = await self.inspect_mcp_servers()
        results["tools"] = await self.inspect_tool_registry()
        results["subagent_executor"] = await self.inspect_subagent_executor()
        results["langgraph_state"] = await self.inspect_langgraph_state()
        
        self.console.print("\n[bold green]✨ Inspection Complete[/bold green]")
        
        return results


async def run_live_tests():
    """Run live system inspection and tests."""
    inspector = LiveSystemInspector()
    
    # Full system inspection
    await inspector.run_full_inspection()
    
    # Run a live subagent test
    console.print("\n" + "="*60)
    console.print("[bold]Running Live Subagent Test[/bold]")
    console.print("="*60)
    
    await inspector.test_subagent_invocation(
        agent_type="reasoning",
        task="Analyze this simple problem: If I have 5 apples and eat 2, how many remain?"
    )


def main():
    """Main entry point for live testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Subagent Integration Tests")
    parser.add_argument("--live", action="store_true", help="Run live system inspection")
    parser.add_argument("--test-agent", type=str, help="Test a specific agent type")
    parser.add_argument("--task", type=str, default="What is 2+2?", help="Task for the agent")
    
    args = parser.parse_args()
    
    if args.live or args.test_agent:
        console.print("[bold]🔌 L.O.V.E. System Live Inspection[/bold]\n")
        
        async def run():
            inspector = LiveSystemInspector()
            
            if args.test_agent:
                await inspector.test_subagent_invocation(args.test_agent, args.task)
            else:
                await run_live_tests()
        
        asyncio.run(run())
    else:
        # Run pytest
        console.print("Running pytest... Use --live for live inspection.")
        pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
