#!/usr/bin/env python3
"""
MCP Integration Test - Story 4: End-to-End Verification

This test verifies the complete MCP tool integration pipeline:
1. MCPManager lifecycle
2. MCP-to-LangChain tool adapter
3. ToolRetriever integration with MCP tools
4. Actual tool execution (optional, requires GITHUB_PERSONAL_ACCESS_TOKEN)

Usage:
    python tests/verify_mcp_flow.py [--live]

The --live flag enables actual GitHub API calls (requires token).
Without it, only the adapter and retriever integration are tested.
"""

import os
import sys
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_shared_state_has_mcp_manager_field():
    """Test that shared_state.py has the mcp_manager field."""
    console.print("[bold]Test 1: shared_state has mcp_manager field[/bold]")
    
    import core.shared_state as shared_state
    
    assert hasattr(shared_state, 'mcp_manager'), "shared_state missing 'mcp_manager' field"
    console.print("[green]✓ shared_state.mcp_manager field exists[/green]")
    return True


def test_mcp_adapter_imports():
    """Test that the MCP adapter module can be imported."""
    console.print("\n[bold]Test 2: MCP adapter imports correctly[/bold]")
    
    from core.mcp_adapter import (
        convert_mcp_to_langchain_tools,
        get_all_mcp_langchain_tools,
        _create_pydantic_model_from_schema,
        JSON_SCHEMA_TYPE_MAP
    )
    
    console.print("[green]✓ core.mcp_adapter imports successfully[/green]")
    return True


def test_pydantic_model_creation():
    """Test dynamic Pydantic model creation from JSON Schema."""
    console.print("\n[bold]Test 3: Pydantic model creation from JSON Schema[/bold]")
    
    from core.mcp_adapter import _create_pydantic_model_from_schema
    
    test_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
            "include_archived": {"type": "boolean", "description": "Include archived repos"}
        },
        "required": ["query"]
    }
    
    model = _create_pydantic_model_from_schema("TestTool", test_schema)
    
    assert model is not None, "Model creation failed"
    assert model.__name__ == "TestToolInput", f"Expected 'TestToolInput', got '{model.__name__}'"
    
    # Verify fields exist
    field_names = list(model.model_fields.keys())
    assert "query" in field_names, "Missing 'query' field"
    assert "limit" in field_names, "Missing 'limit' field"
    
    console.print(f"[green]✓ Created model with fields: {field_names}[/green]")
    return True


def test_mcp_manager_has_list_tools():
    """Test that MCPManager has the list_tools method."""
    console.print("\n[bold]Test 4: MCPManager has list_tools method[/bold]")
    
    from mcp_manager import MCPManager
    
    assert hasattr(MCPManager, 'list_tools'), "MCPManager missing 'list_tools' method"
    
    # Create a manager instance and test the method
    manager = MCPManager(console)
    
    # Should return tools from config
    github_tools = manager.list_tools("github")
    assert isinstance(github_tools, dict), "list_tools should return a dict"
    
    if github_tools:
        console.print(f"[green]✓ list_tools('github') returned {len(github_tools)} tools[/green]")
        for tool_name in list(github_tools.keys())[:3]:
            console.print(f"  - {tool_name}")
    else:
        console.print("[yellow]⚠ No tools defined for 'github' in mcp_servers.json[/yellow]")
    
    return True


def test_tool_retriever_integration():
    """Test that ToolRetriever can be initialized for semantic tool search."""
    console.print("\n[bold]Test 5: ToolRetriever integration[/bold]")
    
    from core.nodes.tool_retrieval import get_retriever, refresh_mcp_tools
    from core.tool_retriever import ToolRetriever
    
    # Get the retriever (will attempt to load MCP tools if available)
    retriever = get_retriever()
    
    assert retriever is not None, "Failed to get retriever"
    assert isinstance(retriever, ToolRetriever), "Retriever should be a ToolRetriever instance"
    assert hasattr(retriever, 'retrieve'), "Retriever missing 'retrieve' method"
    assert hasattr(retriever, 'index_tools'), "Retriever missing 'index_tools' method"
    
    console.print(f"[green]✓ ToolRetriever initialized with {len(retriever._tool_cache)} cached tools[/green]")
    
    # Test refresh function exists
    assert callable(refresh_mcp_tools), "refresh_mcp_tools should be callable"
    console.print("[green]✓ refresh_mcp_tools function available[/green]")
    
    return True


def test_adapter_creates_langchain_tools():
    """Test that the adapter creates proper LangChain tools from MCP definitions."""
    console.print("\n[bold]Test 6: Adapter creates LangChain tools[/bold]")
    
    from mcp_manager import MCPManager
    from core.mcp_adapter import convert_mcp_to_langchain_tools
    from langchain_core.tools import BaseTool
    
    manager = MCPManager(console)
    
    # Convert GitHub MCP tools to LangChain format
    tools = convert_mcp_to_langchain_tools("github", manager)
    
    if not tools:
        console.print("[yellow]⚠ No tools converted (GitHub might not be configured)[/yellow]")
        return True
    
    console.print(f"[green]✓ Converted {len(tools)} MCP tools to LangChain format[/green]")
    
    for tool in tools[:3]:
        assert isinstance(tool, BaseTool), f"Tool {tool.name} is not a BaseTool"
        console.print(f"  - {tool.name}: {tool.description[:50]}...")
    
    return True


# ============================================================================
# Dynamic Discovery Tests (Epic: MCP Dynamic Discovery)
# ============================================================================

def test_dynamic_discovery_imports():
    """Test that the new dynamic discovery module can be imported."""
    console.print("\n[bold]Test 7: Dynamic discovery module imports[/bold]")
    
    from core.mcp_dynamic_discovery import (
        MCPDynamicDiscovery,
        get_discovery,
        reset_discovery
    )
    
    console.print("[green]✓ core.mcp_dynamic_discovery imports successfully[/green]")
    return True


def test_dynamic_discovery_servers():
    """Test that dynamic discovery can list servers."""
    console.print("\n[bold]Test 8: Dynamic discovery lists servers[/bold]")
    
    from core.mcp_dynamic_discovery import MCPDynamicDiscovery
    from mcp_manager import MCPManager
    
    manager = MCPManager(console)
    discovery = MCPDynamicDiscovery(mcp_manager=manager)
    
    servers = discovery.discover_servers()
    
    assert isinstance(servers, list), "discover_servers should return a list"
    console.print(f"[green]✓ discover_servers returned {len(servers)} servers[/green]")
    
    if servers:
        for s in servers:
            console.print(f"  - {s['name']}: {s['tool_count']} tools [{s['status']}]")
    
    return True


def test_dynamic_discovery_tools():
    """Test that dynamic discovery can list tools for a server."""
    console.print("\n[bold]Test 9: Dynamic discovery lists tools[/bold]")
    
    from core.mcp_dynamic_discovery import MCPDynamicDiscovery
    from mcp_manager import MCPManager
    
    manager = MCPManager(console)
    discovery = MCPDynamicDiscovery(mcp_manager=manager)
    
    tools = discovery.discover_tools("github")
    
    assert isinstance(tools, dict), "discover_tools should return a dict"
    
    if "error" not in tools:
        console.print(f"[green]✓ discover_tools('github') returned {len(tools)} tools[/green]")
        for name in list(tools.keys())[:3]:
            console.print(f"  - {name}: {tools[name][:40]}...")
    else:
        console.print(f"[yellow]⚠ {tools['error']}[/yellow]")
    
    return True


def test_meta_tools_registration():
    """Test that dynamic MCP meta-tools can be registered."""
    console.print("\n[bold]Test 10: Meta-tools registration[/bold]")
    
    from core.legacy_compat import ToolRegistry
    from core.mcp_tools import register_dynamic_mcp_tools
    from mcp_manager import MCPManager
    import core.shared_state as shared_state
    
    # Set up mcp_manager in shared_state
    manager = MCPManager(console)
    shared_state.mcp_manager = manager
    
    # Create a fresh registry
    registry = ToolRegistry()
    
    # Register dynamic tools
    registered = register_dynamic_mcp_tools(registry, manager)
    
    assert len(registered) == 3, f"Expected 3 meta-tools, got {len(registered)}"
    assert "mcp_list_servers" in registered, "Missing mcp_list_servers"
    assert "mcp_list_tools" in registered, "Missing mcp_list_tools"
    assert "mcp_call" in registered, "Missing mcp_call"
    
    console.print(f"[green]✓ Registered {len(registered)} meta-tools: {', '.join(registered)}[/green]")
    return True


def test_get_tool_schema():
    """Test that tool schema can be retrieved on demand."""
    console.print("\n[bold]Test 11: Get tool schema on demand[/bold]")
    
    from core.mcp_dynamic_discovery import MCPDynamicDiscovery
    from mcp_manager import MCPManager
    
    manager = MCPManager(console)
    discovery = MCPDynamicDiscovery(mcp_manager=manager)
    
    # Get schema for a known tool
    schema = discovery.get_tool_schema("github", "repos.search_repositories")
    
    assert isinstance(schema, dict), "get_tool_schema should return a dict"
    assert "error" not in schema, f"Got error: {schema.get('error')}"
    assert "name" in schema, "Schema should have 'name' field"
    assert "description" in schema, "Schema should have 'description' field"
    assert "parameters" in schema, "Schema should have 'parameters' field"
    
    console.print(f"[green]✓ Retrieved schema for repos.search_repositories[/green]")
    console.print(f"  - Name: {schema['name']}")
    console.print(f"  - Description: {schema['description'][:60]}...")
    
    return True


def test_discovery_error_handling():
    """Test error handling for non-existent servers/tools."""
    console.print("\n[bold]Test 12: Error handling for non-existent servers/tools[/bold]")
    
    from core.mcp_dynamic_discovery import MCPDynamicDiscovery
    from mcp_manager import MCPManager
    
    manager = MCPManager(console)
    discovery = MCPDynamicDiscovery(mcp_manager=manager)
    
    # Test non-existent server
    tools = discovery.discover_tools("nonexistent_server")
    assert "error" in tools, "Should return error for non-existent server"
    console.print(f"[green]✓ Non-existent server returns error: {tools['error']}[/green]")
    
    # Test non-existent tool
    schema = discovery.get_tool_schema("github", "nonexistent_tool")
    assert "error" in schema, "Should return error for non-existent tool"
    console.print(f"[green]✓ Non-existent tool returns error: {schema['error']}[/green]")
    
    return True


def test_execute_tool_method():
    """Test that execute_tool method works correctly."""
    console.print("\n[bold]Test 13: Execute tool method structure[/bold]")
    
    from core.mcp_dynamic_discovery import MCPDynamicDiscovery
    from mcp_manager import MCPManager
    
    manager = MCPManager(console)
    discovery = MCPDynamicDiscovery(mcp_manager=manager)
    
    # Verify method exists and has correct signature
    assert hasattr(discovery, 'execute_tool'), "Discovery should have execute_tool method"
    
    import inspect
    sig = inspect.signature(discovery.execute_tool)
    params = list(sig.parameters.keys())
    
    assert "server_name" in params, "execute_tool should accept server_name"
    assert "tool_name" in params, "execute_tool should accept tool_name"
    assert "params" in params, "execute_tool should accept params"
    
    console.print(f"[green]✓ execute_tool has correct signature: {list(params)}[/green]")
    
    # Test without starting server (should fail gracefully)
    result = discovery.execute_tool("github", "repos.search_repositories", {"query": "test"})
    # Should return an error message (server not running)
    assert isinstance(result, str), "execute_tool should return a string"
    console.print(f"[green]✓ execute_tool returns string result[/green]")
    
    return True


def test_discovery_singleton():
    """Test that the discovery singleton pattern works correctly."""
    console.print("\n[bold]Test 14: Discovery singleton pattern[/bold]")
    
    from core.mcp_dynamic_discovery import get_discovery, reset_discovery, MCPDynamicDiscovery
    
    # Reset first to ensure clean state
    reset_discovery()
    
    # Get instance
    instance1 = get_discovery()
    assert isinstance(instance1, MCPDynamicDiscovery), "get_discovery should return MCPDynamicDiscovery"
    
    # Get again - should be same instance
    instance2 = get_discovery()
    assert instance1 is instance2, "get_discovery should return same singleton instance"
    
    console.print("[green]✓ Singleton pattern works correctly[/green]")
    
    # Test reset
    reset_discovery()
    instance3 = get_discovery()
    assert instance3 is not instance1, "After reset, should get new instance"
    
    console.print("[green]✓ reset_discovery creates new instance[/green]")
    
    return True


async def test_live_github_search():
    """
    Live test: Actually calls the GitHub MCP server.
    Requires GITHUB_PERSONAL_ACCESS_TOKEN environment variable.
    """
    console.print("\n[bold]Test 7 (LIVE): GitHub MCP tool execution[/bold]")
    
    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        console.print("[yellow]⚠ GITHUB_PERSONAL_ACCESS_TOKEN not set - skipping live test[/yellow]")
        return True
    
    # Check Docker
    import subprocess
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            console.print("[yellow]⚠ Docker not available - skipping live test[/yellow]")
            return True
    except:
        console.print("[yellow]⚠ Docker not found - skipping live test[/yellow]")
        return True
    
    from mcp_manager import MCPManager
    import core.shared_state as shared_state
    
    console.print("[cyan]Starting GitHub MCP server...[/cyan]")
    
    manager = MCPManager(console)
    shared_state.mcp_manager = manager
    
    result = manager.start_server("github", env_vars={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token})
    
    if "successfully" not in result.lower():
        console.print(f"[red]✗ Failed to start server: {result}[/red]")
        return False
    
    console.print(f"[green]✓ {result}[/green]")
    
    # Wait for server to be ready
    console.print("[cyan]Waiting for server to be ready...[/cyan]")
    await asyncio.sleep(5)
    
    try:
        # Test calling a tool
        console.print("[cyan]Calling repos.search_repositories tool...[/cyan]")
        
        request_id = manager.call_tool("github", "repos.search_repositories", {"query": "linux"})
        response = manager.get_response("github", request_id, timeout=30)
        
        if "error" in response:
            console.print(f"[red]✗ Tool error: {response['error']}[/red]")
            return False
        
        console.print("[green]✓ Tool call returned successfully[/green]")
        
        # Check if 'torvalds' appears in the result
        result_str = str(response.get("result", ""))
        if "torvalds" in result_str.lower():
            console.print("[green]✓ Result contains 'torvalds' (expected owner of linux repo)[/green]")
        else:
            console.print("[yellow]⚠ 'torvalds' not found in result (might be ok depending on search results)[/yellow]")
        
        return True
        
    finally:
        console.print("[cyan]Stopping MCP server...[/cyan]")
        manager.stop_all_servers()
        console.print("[green]✓ Server stopped[/green]")


def main():
    """Run all MCP integration tests."""
    console.print(Panel.fit(
        "[bold magenta]MCP Tool Integration Verification[/bold magenta]\n"
        "Testing the complete MCP-to-LangChain pipeline",
        border_style="magenta"
    ))
    
    live_mode = "--live" in sys.argv
    
    tests = [
        ("shared_state field", test_shared_state_has_mcp_manager_field),
        ("MCP adapter imports", test_mcp_adapter_imports),
        ("Pydantic model creation", test_pydantic_model_creation),
        ("MCPManager list_tools", test_mcp_manager_has_list_tools),
        ("ToolRetriever integration", test_tool_retriever_integration),
        ("Adapter creates LangChain tools", test_adapter_creates_langchain_tools),
        # Dynamic Discovery Tests (Epic: MCP Dynamic Discovery)
        ("Dynamic discovery imports", test_dynamic_discovery_imports),
        ("Dynamic discovery servers", test_dynamic_discovery_servers),
        ("Dynamic discovery tools", test_dynamic_discovery_tools),
        ("Meta-tools registration", test_meta_tools_registration),
        ("Get tool schema", test_get_tool_schema),
        ("Discovery error handling", test_discovery_error_handling),
        ("Execute tool method", test_execute_tool_method),
        ("Discovery singleton", test_discovery_singleton),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            console.print(f"[red]✗ {name} FAILED: {e}[/red]")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Live test (only with --live flag)
    if live_mode:
        try:
            if asyncio.run(test_live_github_search()):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            console.print(f"[red]✗ Live GitHub test FAILED: {e}[/red]")
            failed += 1
    else:
        console.print("\n[dim]Run with --live flag to test actual GitHub MCP server[/dim]")
    
    console.print("\n" + "=" * 50)
    console.print(f"[bold]Results: {passed} passed, {failed} failed[/bold]")
    
    if failed == 0:
        console.print("[bold green]All tests passed! ✓[/bold green]")
        return 0
    else:
        console.print("[bold red]Some tests failed ✗[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
