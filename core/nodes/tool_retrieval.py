from typing import Dict, Any, List
from core.state import DeepAgentState
from core.tool_retriever import ToolRetriever
from core.tools import (
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve
)

ALL_TOOLS = [
    execute, decompose_and_solve_subgoal, evolve, post_to_bluesky,
    read_file, write_file, scan_network, probe_target,
    perform_webrequest, analyze_json_file, research_and_evolve
]

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = ToolRetriever(ALL_TOOLS)
    return _retriever

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
