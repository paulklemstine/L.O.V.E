from typing import Dict, Any, List
from core.state import DeepAgentState
from core.tool_retriever import ToolRetriever
from core.tools import (
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve,
    speak_to_creator
)

ALL_TOOLS = [
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve,
    speak_to_creator
]

_retriever = None

def get_retriever():
    """
    Gets the ToolRetriever singleton, initializing it with both native tools
    and any available MCP tools from running servers.
    """
    global _retriever
    if _retriever is None:
        _retriever = ToolRetriever(ALL_TOOLS)
        
        # Dynamically add MCP tools if available
        _load_mcp_tools()
    
    return _retriever


def _load_mcp_tools():
    """
    Loads MCP tools from running servers and adds them to the retriever.
    Called once during retriever initialization.
    """
    global _retriever
    if _retriever is None:
        return
    
    try:
        import core.shared_state as shared_state
        from core.mcp_adapter import get_all_mcp_langchain_tools
        import core.logging
        
        if shared_state.mcp_manager is None:
            return
        
        # Get all MCP tools as LangChain tools
        mcp_tools = get_all_mcp_langchain_tools(shared_state.mcp_manager)
        
        if mcp_tools:
            _retriever.add_tools(mcp_tools)
            core.logging.log_event(
                f"Added {len(mcp_tools)} MCP tools to ToolRetriever for semantic search",
                "INFO"
            )
    except ImportError as e:
        # MCP adapter not available
        pass
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
    global _retriever
    if _retriever is not None:
        _load_mcp_tools()


async def retrieve_tools_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Node responsible for retrieving relevant tools based on a query.
    """
    query = state.get("tool_query")
    if not query:
        return {}
        
    retriever = get_retriever()
    # Assuming query is a string. If it's a list or something else, handle it.
    tools = retriever.query_tools(query)
    
    return {"retrieved_tools": tools}

