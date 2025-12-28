"""
DeepAgent Graph Definition.

This module constructs the main LangGraph StateGraph for the DeepAgent system,
implementing the ReAct pattern with self-correction loops.

Key Features:
- Supervisor routing to specialized teams
- Reasoning node with tool call detection
- Self-correction loop: tool_execution -> reasoning (up to MAX_ITERATIONS)
- Memory folding and tool retrieval integration
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from core.state import DeepAgentState
from core.nodes.supervisor import supervisor_node
from core.nodes.reasoning import reason_node
from core.nodes.execution import tool_execution_node
from core.nodes.tool_retrieval import retrieve_tools_node
from core.nodes.memory import fold_memory_node
from core.nodes.social_media_team import social_media_node
from core.nodes.evolution_team import evolution_node
from core.graphs.coding_team import create_coding_graph

# Maximum iterations for the ReAct loop before forcing termination
MAX_ITERATIONS = 5


def create_deep_agent_graph():
    """
    Constructs the main DeepAgent StateGraph.
    
    Architecture:
    - Entry: supervisor
    - Supervisor routes to: coding_team, reasoning_node, social_media_team, evolution_team
    - Reasoning node routes to: tool_execution_node, fold_memory_node, retrieve_tools_node, or END
    - Self-correction loop: tool_execution_node -> reasoning_node (capped at MAX_ITERATIONS)
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
        """Routes from supervisor to appropriate team or reasoning node."""
        next_node = state.get("next_node")
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
            "evolution_team": "evolution_team",
            "reasoning_node": "reasoning_node"
        }
    )
    
    # Reasoning Routing (Story 2 & 4 implementation)
    def route_reasoning(state: DeepAgentState):
        """
        Routes based on reasoning node output.
        
        Implements the ReAct pattern:
        - tool_call/tool_calls in response -> route to tool_execution_node
        - fold_thought control token -> route to fold_memory_node
        - retrieve_tool control token -> route to retrieve_tools_node
        - None/text response -> route to END
        
        Also enforces MAX_ITERATIONS guardrail for self-correction loop.
        """
        stop_reason = state.get("stop_reason")
        loop_count = state.get("loop_count", 0)
        
        # Story 4: Guardrail - Check max iterations
        if loop_count >= MAX_ITERATIONS:
            # Force END to prevent infinite loops
            return END
        
        # Route based on stop_reason
        if stop_reason == "fold_thought":
            return "fold_memory_node"
        elif stop_reason == "retrieve_tool":
            return "retrieve_tools_node"
        elif stop_reason == "tool_call":
            # Story 2: Route to tool execution when tool_calls detected
            return "tool_execution_node"
            
        # Legacy: Check for tool calls in messages (backward compatibility)
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tool_execution_node"
        
        # No tool calls, no control tokens -> direct response, end
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
    
    # Story 4: Self-Correction Loop
    # Tool execution routes BACK to reasoning for observation synthesis
    # The reasoning node will see: [User Request] -> [Agent Tool Call] -> [Tool Output]
    # And can decide to try again, call another tool, or provide final response
    workflow.add_edge("tool_execution_node", "reasoning_node")
    
    # Memory and tool retrieval also loop back to reasoning
    workflow.add_edge("retrieve_tools_node", "reasoning_node")
    workflow.add_edge("fold_memory_node", "reasoning_node")
    
    # Team nodes route back to supervisor
    workflow.add_edge("coding_team", "supervisor") 
    workflow.add_edge("social_media_team", "supervisor") 
    workflow.add_edge("evolution_team", "tool_execution_node") 
    
    return workflow.compile()

