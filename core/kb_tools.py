"""
Knowledge Base Tools for DeepAgent

Provides tools for querying the knowledge graph and searching memories.
"""
import json
from typing import Optional, List, Dict, Any


def query_knowledge_base(node_type: str, limit: int = 10, knowledge_base=None) -> str:
    """
    Queries the main knowledge graph by node type.
    
    Args:
        node_type: Type of nodes to query (e.g., 'task', 'talent', 'host', 'opportunity')
        limit: Maximum number of results to return
        knowledge_base: GraphDataManager instance
        
    Returns:
        JSON string with query results
    """
    if not knowledge_base:
        return json.dumps({"error": "Knowledge base not available"})
    
    try:
        nodes = knowledge_base.query_nodes("node_type", node_type)
        results = []
        
        for node_id in nodes[:limit]:
            node_data = knowledge_base.get_node(node_id)
            if node_data:
                # Clean up the data for JSON serialization
                clean_data = {k: v for k, v in node_data.items() 
                             if isinstance(v, (str, int, float, bool, list, dict))}
                results.append({
                    "id": node_id,
                    "type": node_type,
                    "data": clean_data
                })
        
        return json.dumps({
            "node_type": node_type,
            "count": len(results),
            "results": results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Query failed: {str(e)}"})


def search_memories(query: str, top_k: int = 3, memory_manager=None) -> str:
    """
    Performs semantic search on the memory system using FAISS.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        memory_manager: MemoryManager instance
        
    Returns:
        JSON string with relevant memories
    """
    if not memory_manager:
        return json.dumps({"error": "Memory manager not available"})
    
    try:
        # Use the existing retrieve_relevant_folded_memories method
        memories = memory_manager.retrieve_relevant_folded_memories(query, top_k=top_k)
        
        if not memories:
            return json.dumps({
                "query": query,
                "count": 0,
                "memories": []
            })
        
        return json.dumps({
            "query": query,
            "count": len(memories),
            "memories": memories
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Memory search failed: {str(e)}"})


def get_kb_summary(knowledge_base=None, max_tokens: int = 512) -> str:
    """
    Returns a high-level summary of the knowledge base.
    
    Args:
        knowledge_base: GraphDataManager instance
        max_tokens: Maximum tokens for the summary
        
    Returns:
        Text summary of the knowledge base
    """
    if not knowledge_base:
        return "Knowledge base not available"
    
    try:
        summary, nodes_by_type = knowledge_base.summarize_graph(max_tokens=max_tokens)
        return summary
    
    except Exception as e:
        return f"Error generating KB summary: {str(e)}"


def get_active_tasks(knowledge_base=None) -> str:
    """
    Returns a list of active tasks from the knowledge base.
    
    Args:
        knowledge_base: GraphDataManager instance
        
    Returns:
        JSON string with active tasks
    """
    if not knowledge_base:
        return json.dumps({"error": "Knowledge base not available"})
    
    try:
        task_nodes = knowledge_base.query_nodes("node_type", "task")
        tasks = []
        
        for node_id in task_nodes:
            node_data = knowledge_base.get_node(node_id)
            if node_data:
                tasks.append({
                    "id": node_id,
                    "status": node_data.get("status", "unknown"),
                    "request": node_data.get("request", "No description")[:200]
                })
        
        return json.dumps({
            "count": len(tasks),
            "tasks": tasks
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Failed to get tasks: {str(e)}"})
