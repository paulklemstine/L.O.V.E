from typing import TypedDict, Annotated, List, Any, Optional, Dict
from langgraph.graph.message import add_messages
from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory

class DeepAgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    episodic_memory: EpisodicMemory
    working_memory: WorkingMemory
    tool_memory: ToolMemory
    next_node: Optional[str]
    recursion_depth: int
    stop_reason: Optional[str]
    # Add fields for tool retrieval
    tool_query: Optional[str]
    retrieved_tools: List[Any]
    # Creator Interaction
    creator_mandate: Optional[str]
    # Tool Registry Integration (Story 2)
    tool_schemas: List[Dict[str, Any]]  # JSON schemas for LLM tool binding
    # Self-Correction Loop Guardrail (Story 4)
    loop_count: int  # Tracks reasoning-execution cycles, max 5

