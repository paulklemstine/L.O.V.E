from core.nodes.evolution_team import evolution_node

def create_deep_agent_graph():
    """
    Constructs the main DeepAgent StateGraph.
    """
    workflow = StateGraph(DeepAgentState)
    
    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reasoning_node", reason_node)
    workflow.add_node("tool_execution_node", tool_execution_node)
    workflow.add_node("retrieve_tools_node", retrieve_tools_node)
    workflow.add_node("fold_memory_node", fold_memory_node)
    workflow.add_node("social_media_team", social_media_node)
    workflow.add_node("evolution_team", evolution_node)
    
    # Add Subgraphs
    coding_graph = create_coding_graph()
    workflow.add_node("coding_team", coding_graph)
    
    # Entry Point
    workflow.set_entry_point("supervisor")
    
    # Supervisor Routing
    def route_supervisor(state: DeepAgentState):
        next_node = state.get("next_node")
        # Removing fallback for evolution_team as it is now implemented
        if next_node in ["coding_team", "reasoning_node", "social_media_team", "evolution_team"]:
            return next_node
        # Default fallback
        return "reasoning_node"
        
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "coding_team": "coding_team",
            "social_media_team": "social_media_team",
            "reasoning_node": "reasoning_node"
        }
    )
    
    # Reasoning Routing
    def route_reasoning(state: DeepAgentState):
        stop_reason = state.get("stop_reason")
        if stop_reason == "fold_thought":
            return "fold_memory_node"
        elif stop_reason == "retrieve_tool":
            return "retrieve_tools_node"
            
        # Check for tool calls
        # Assuming reason_node or LLM adds tool_calls to the last message
        messages = state["messages"]
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tool_execution_node"
                
        return END
        
    workflow.add_conditional_edges(
        "reasoning_node",
        route_reasoning,
        {
            "fold_memory_node": "fold_memory_node",
            "retrieve_tools_node": "retrieve_tools_node",
            "tool_execution_node": "tool_execution_node",
            END: END
        }
    )
    
    # Edges returning to reasoning or supervisor
    workflow.add_edge("tool_execution_node", "reasoning_node")
    workflow.add_edge("retrieve_tools_node", "reasoning_node")
    workflow.add_edge("fold_memory_node", "reasoning_node")
    workflow.add_edge("coding_team", "supervisor") 
    workflow.add_edge("social_media_team", "supervisor") 
    workflow.add_edge("evolution_team", "tool_execution_node") 
    
    return workflow.compile()
