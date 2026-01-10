from typing import Dict, Any, List
from core.state import DeepAgentState
from core.tool_retriever import get_tool_retriever, ToolRetriever


_retriever_initialized = False


def get_retriever() -> ToolRetriever:
    """
    Gets the ToolRetriever singleton, initializing it with both native tools
    and any available MCP tools from running servers.
    """
    global _retriever_initialized
    retriever = get_tool_retriever()
    
    if not _retriever_initialized:
        # Try to initialize with tool registry
        _initialize_retriever(retriever)
        # Load MCP tools if available
        _load_mcp_tools(retriever)
        _retriever_initialized = True
    
    return retriever


def _initialize_retriever(retriever: ToolRetriever):
    """
    Initialize the retriever with the global tool registry.
    """
    try:
        import core.shared_state as shared_state
        if shared_state.tool_registry:
            retriever.index_tools(shared_state.tool_registry)
    except Exception as e:
        import core.logging
        core.logging.log_event(f"Could not initialize ToolRetriever: {e}", "WARNING")


def _load_mcp_tools(retriever: ToolRetriever):
    """
    Loads MCP tools from running servers by re-indexing the registry.
    The MCP tools should already be registered in the tool registry.
    """
    try:
        import core.shared_state as shared_state
        import core.logging
        
        if shared_state.mcp_manager is None:
            return
        
        # Re-index tools to pick up any MCP tools added to registry
        if shared_state.tool_registry:
            retriever.index_tools(shared_state.tool_registry)
            core.logging.log_event(
                "ToolRetriever re-indexed to include MCP tools",
                "DEBUG"
            )
    except Exception as e:
        import core.logging
        core.logging.log_event(
            f"Failed to load MCP tools into ToolRetriever: {e}",
            "WARNING"
        )


def refresh_mcp_tools():
    """
    Refreshes MCP tools in the retriever. Call this when MCP servers are started/stopped.
    """
    global _retriever_initialized
    retriever = get_tool_retriever()
    _load_mcp_tools(retriever)


async def retrieve_tools_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Node responsible for retrieving relevant tools based on a query.
    """
    query = state.get("tool_query")
    if not query:
        return {}
        
    retriever = get_retriever()
    
    # Use retrieve() method with the query as step description
    tool_matches = retriever.retrieve(query, max_tools=10)
    
    # Convert ToolMatch objects to tool references
    tools = [match.name for match in tool_matches]
    
    return {"retrieved_tools": tools}

