"""
Dynamic Tooling Tools - Interface for L.O.V.E. to interact with CodeAct, MCP Registry, etc.

Exposes the Open Agentic Web capabilities to the agent:
- Search and install MCP servers
- Generate new MCP servers  
- Manage skill library
- Run code in a safe sandbox
"""

import asyncio
from typing import Optional, List, Dict, Any

from core.tool_registry import tool_schema, ToolResult


# =============================================================================
# MCP Registry Tools
# =============================================================================

@tool_schema
def search_mcp_servers(query: str, limit: int = 5) -> ToolResult:
    """
    Search public MCP registries for servers matching a capability.
    
    Searches across mcp.so, Smithery.ai, and GitHub registries.
    
    Args:
        query: What capability to search for (e.g., "weather", "database", "github")
        limit: Maximum number of results to return
        
    Returns:
        List of matching MCP servers with installation info
    """
    from core.mcp_registry import get_mcp_registry
    
    registry = get_mcp_registry()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        servers = loop.run_until_complete(registry.search(query, limit=limit))
        
        # Convert to serializable format
        results = []
        for server in servers:
            results.append({
                "id": server.id,
                "name": server.name,
                "description": server.description[:200] if server.description else "",
                "author": server.author,
                "registry": server.registry,
                "npm_package": server.npm_package,
                "github_url": server.github_url
            })
        
        if results:
            return ToolResult(
                status="success",
                data={"servers": results, "count": len(results)},
                observation=f"Found {len(results)} MCP servers for '{query}':\n" + 
                           "\n".join(f"  - {s['name']}: {s['description'][:80]}..." for s in results)
            )
        else:
            return ToolResult(
                status="success",
                data={"servers": [], "count": 0},
                observation=f"No MCP servers found for '{query}'. Try different keywords."
            )
            
    except Exception as e:
        return ToolResult(
            status="error",
            data={"error": str(e)},
            observation=f"Failed to search MCP registries: {e}"
        )


@tool_schema 
def install_mcp_server(server_id: str, registry: str = "mcp.so") -> ToolResult:
    """
    Install an MCP server from a public registry.
    
    The server will be downloaded and configured for local use.
    
    Args:
        server_id: The ID of the server to install (from search results)
        registry: Which registry to install from (mcp.so, smithery, github)
        
    Returns:
        Installation result with path and configuration
    """
    from core.mcp_registry import get_mcp_registry, MCPServerInfo
    
    mcp_registry = get_mcp_registry()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # First search to get the server info
    try:
        servers = loop.run_until_complete(mcp_registry.search(server_id, registries=[registry], limit=5))
        
        # Find matching server
        server = None
        for s in servers:
            if s.id == server_id or server_id in s.name.lower():
                server = s
                break
        
        if not server:
            return ToolResult(
                status="error",
                data={"error": "Server not found"},
                observation=f"Server '{server_id}' not found in {registry}. Use search_mcp_servers first."
            )
        
        # Install
        success, message = loop.run_until_complete(mcp_registry.install(server))
        
        if success:
            return ToolResult(
                status="success",
                data={"server": server.to_dict(), "message": message},
                observation=f"Successfully installed MCP server '{server.name}': {message}"
            )
        else:
            return ToolResult(
                status="error",
                data={"error": message},
                observation=f"Failed to install server: {message}"
            )
            
    except Exception as e:
        return ToolResult(
            status="error",
            data={"error": str(e)},
            observation=f"Installation failed: {e}"
        )


@tool_schema
def list_installed_mcp_servers() -> ToolResult:
    """
    List all locally installed MCP servers.
    
    Returns:
        List of installed servers with their configurations
    """
    from core.mcp_registry import get_mcp_registry
    
    registry = get_mcp_registry()
    installed = registry.list_installed()
    
    if installed:
        return ToolResult(
            status="success",
            data={"servers": installed, "count": len(installed)},
            observation=f"Installed MCP servers ({len(installed)}):\n" +
                       "\n".join(f"  - {s['name']}" for s in installed)
        )
    else:
        return ToolResult(
            status="success",
            data={"servers": [], "count": 0},
            observation="No MCP servers installed. Use search_mcp_servers and install_mcp_server."
        )


# =============================================================================
# MCP Server Synthesis Tools
# =============================================================================

@tool_schema
def synthesize_mcp_server(capability_description: str, server_name: Optional[str] = None) -> ToolResult:
    """
    Generate a new MCP server for a novel capability.
    
    Uses the Evolutionary Agent to synthesize a complete MCP server
    with Python code, requirements, and Dockerfile.
    
    Args:
        capability_description: What the server should do (e.g., "Fetch cryptocurrency prices")
        server_name: Optional name for the server
        
    Returns:
        Generated server details with file paths
    """
    from core.agents.evolutionary_agent import get_evolutionary_agent
    
    agent = get_evolutionary_agent()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            agent.synthesize_mcp_server(capability_description, server_name)
        )
        
        if result["success"]:
            return ToolResult(
                status="success",
                data=result,
                observation=f"Generated MCP server '{result['server_name']}' at {result['file_path']}"
            )
        else:
            return ToolResult(
                status="error",
                data=result,
                observation=f"Failed to generate server: {result.get('message', 'Unknown error')}"
            )
            
    except Exception as e:
        return ToolResult(
            status="error",
            data={"error": str(e)},
            observation=f"Server synthesis failed: {e}"
        )


# =============================================================================
# Skill Library Tools (Voyager Pattern)
# =============================================================================

@tool_schema
def save_skill(skill_name: str, skill_code: str, description: str, tags: str = "") -> ToolResult:
    """
    Save a skill to the persistent skill library for future reuse.
    
    Part of the Voyager pattern - accumulate successful code for later retrieval.
    
    Args:
        skill_name: Name for the skill (snake_case recommended)
        skill_code: The Python code that implements the skill
        description: What the skill does
        tags: Comma-separated tags for categorization
        
    Returns:
        Confirmation of skill being saved
    """
    from core.agents.evolutionary_agent import get_evolutionary_agent
    
    agent = get_evolutionary_agent()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    success = loop.run_until_complete(
        agent.add_to_skill_library(skill_name, skill_code, description, tag_list)
    )
    
    if success:
        return ToolResult(
            status="success",
            data={"skill_name": skill_name, "tags": tag_list},
            observation=f"Skill '{skill_name}' saved to library with tags: {tag_list}"
        )
    else:
        return ToolResult(
            status="error",
            data={"error": "Failed to save"},
            observation=f"Failed to save skill '{skill_name}' to library"
        )


@tool_schema
def find_skills(task_description: str, max_skills: int = 5) -> ToolResult:
    """
    Find relevant skills from the library for a task.
    
    Searches the accumulated skill library for code that might help
    with the current task.
    
    Args:
        task_description: What you're trying to accomplish
        max_skills: Maximum number of skills to return
        
    Returns:
        List of relevant skills with their code
    """
    from core.agents.evolutionary_agent import get_evolutionary_agent
    
    agent = get_evolutionary_agent()
    skills = agent.get_relevant_skills(task_description, max_skills)
    
    if skills:
        return ToolResult(
            status="success",
            data={"skills": skills, "count": len(skills)},
            observation=f"Found {len(skills)} relevant skills:\n" +
                       "\n".join(f"  - {s['name']}: {s['description']}" for s in skills)
        )
    else:
        return ToolResult(
            status="success",
            data={"skills": [], "count": 0},
            observation="No matching skills found in library. Use save_skill to add new skills."
        )


# =============================================================================
# Docker Sandbox Tools
# =============================================================================

@tool_schema
def run_in_sandbox(code: str, packages: str = "", timeout: int = 30) -> ToolResult:
    """
    Execute Python code in a safe Docker sandbox.
    
    Uses Docker containers when available for isolated execution.
    
    Args:
        code: Python code to execute
        packages: Comma-separated pip packages to install (e.g., "requests,pandas")
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result with stdout/stderr
    """
    from core.docker_sandbox import get_sandbox
    
    sandbox = get_sandbox()
    package_list = [p.strip() for p in packages.split(",") if p.strip()] if packages else None
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(sandbox.run_python(code, packages=package_list))
    
    if result.success:
        return ToolResult(
            status="success",
            data={"stdout": result.stdout, "execution_time": result.execution_time},
            observation=f"Sandbox execution succeeded:\n{result.stdout}"
        )
    else:
        return ToolResult(
            status="error",
            data={"stderr": result.stderr, "exit_code": result.exit_code},
            observation=f"Sandbox execution failed: {result.stderr}"
        )


@tool_schema
def check_docker_available() -> ToolResult:
    """
    Check if Docker is available for sandboxed execution.
    
    Returns:
        Docker availability status and install instructions if needed
    """
    from core.docker_sandbox import DockerSandbox
    
    sandbox = DockerSandbox()
    available = sandbox.is_available()
    
    if available:
        return ToolResult(
            status="success",
            data={"docker_available": True},
            observation="Docker is available. Sandbox execution will use containers."
        )
    else:
        instructions = sandbox.get_install_instructions()
        return ToolResult(
            status="success",
            data={"docker_available": False, "install_instructions": instructions},
            observation=f"Docker not available. Using subprocess fallback. To install: {instructions}"
        )


# =============================================================================
# Tool Discovery
# =============================================================================

@tool_schema
def discover_tools(query: str, max_results: int = 10) -> ToolResult:
    """
    Discover relevant tools using semantic search across all sources.
    
    Searches local registry, MCP servers, and filesystem for matching tools.
    
    Args:
        query: What capability you're looking for
        max_results: Maximum tools to return
        
    Returns:
        Matching tools from all sources
    """
    from core.tool_retriever import get_tool_retriever
    
    retriever = get_tool_retriever()
    
    # Get local matches
    local_matches, mcp_matches = retriever.retrieve_with_mcp(query, max_tools=max_results)
    
    # Get filesystem discovery
    fs_matches = retriever.filesystem_discovery(query)
    
    results = {
        "local_tools": [
            {"name": m.name, "description": m.description, "score": m.score}
            for m in local_matches
        ],
        "mcp_tools": [
            {"name": m.name, "description": m.description, "score": m.score}
            for m in mcp_matches
        ],
        "filesystem_tools": fs_matches
    }
    
    total = len(local_matches) + len(mcp_matches) + len(fs_matches)
    
    observation_parts = []
    if local_matches:
        observation_parts.append(f"Local ({len(local_matches)}): " + 
                                ", ".join(m.name for m in local_matches[:3]))
    if mcp_matches:
        observation_parts.append(f"MCP ({len(mcp_matches)}): " +
                                ", ".join(m.name for m in mcp_matches[:3]))
    if fs_matches:
        observation_parts.append(f"Files ({len(fs_matches)}): " +
                                ", ".join(f['name'] for f in fs_matches[:3]))
    
    return ToolResult(
        status="success",
        data=results,
        observation=f"Found {total} tools for '{query}':\n  " + "\n  ".join(observation_parts)
    )


# =============================================================================
# Registration
# =============================================================================

def register_dynamic_tools():
    """Register all dynamic tooling tools with the global registry."""
    from core.tool_registry import get_global_registry
    
    registry = get_global_registry()
    
    tools = [
        search_mcp_servers,
        install_mcp_server,
        list_installed_mcp_servers,
        synthesize_mcp_server,
        save_skill,
        find_skills,
        run_in_sandbox,
        check_docker_available,
        discover_tools,
    ]
    
    for tool in tools:
        try:
            registry.register(tool)
        except Exception as e:
            print(f"Failed to register {tool.__name__}: {e}")
    
    return len(tools)


# Auto-register when module is imported
_registered = False

def ensure_registered():
    """Ensure tools are registered (idempotent)."""
    global _registered
    if not _registered:
        register_dynamic_tools()
        _registered = True
