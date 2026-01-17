from typing import TypedDict, Annotated, List, Any, Optional, Dict, Tuple
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory


class DeepAgentState(TypedDict):
    """
    State schema for the DeepAgent recursive reasoning loop.
    
    DeepAgent Protocol Story 1.1: Supports recursive reasoning traces
    with Plan→Execute→Critic cycle and self-correction capabilities.
    """
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
    # Ghost Tool & Hallucination Handling
    executed_tool_calls: List[str]  # Track executed tool calls (name+args hash) to prevent duplicates

    # Self-Correction Loop Guardrail (Story 4)
    loop_count: int  # Tracks reasoning-execution cycles, max 5
    # Story 2.1: Semantic Memory Bridge - past interactions context
    memory_context: List[Dict[str, Any]]  # Similar past interactions from FAISS
    # Reference to memory manager for memory operations
    memory_manager: Optional[Any]
    # Subagent invocation tracking
    subagent_results: List[Dict[str, Any]]  # Results from spawned subagents
    subagent_results: List[Dict[str, Any]]  # Results from spawned subagents
    parent_task_id: Optional[str]           # If this is a subagent, parent's ID
    task_id: Optional[str]                  # Unique identifier for this task
    
    # Story 2.3: Theory of Mind Context
    user_model_context: Optional[str]       # Summary of user preferences/beliefs

    # Story 5.5: Shadow Mode Validation (Will Framework)
    shadow_mode: Optional[bool]
    shadow_log: Optional[List[Dict[str, Any]]]
    
    # =========================================================================
    # DeepAgent Protocol - Story 1.1: Recursive Reasoning Trace Fields
    # =========================================================================
    input: Optional[str]  # Original user query/prompt
    chat_history: List[BaseMessage]  # Full conversation history for context
    plan: List[str]  # Numbered atomic steps from PlannerAgent
    past_steps: List[Tuple[str, str, str]]  # (step_description, action_taken, result)
    scratchpad: str  # Working notes for self-correction between loops
    criticism: str  # Critic feedback on last execution attempt
    max_loops: int  # Hard limit on Plan→Execute→Critic cycles (default 5)
    current_loop: int  # Current iteration counter within the cycle


def create_initial_state(
    user_input: str,
    memory_manager: Optional[Any] = None,
    max_loops: int = 5
) -> DeepAgentState:
    """
    Creates a properly initialized DeepAgentState with sensible defaults.
    
    Story 1.1: Helper function to ensure consistent state initialization
    for the recursive reasoning loop.
    
    Args:
        user_input: The original user query/prompt
        memory_manager: Optional reference to memory manager
        max_loops: Maximum Plan→Execute→Critic cycles (default 5)
        
    Returns:
        Initialized DeepAgentState ready for graph execution
    """
    from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory
    
    return DeepAgentState(
        messages=[],
        episodic_memory=EpisodicMemory(),
        working_memory=WorkingMemory(),
        tool_memory=ToolMemory(),
        next_node=None,
        recursion_depth=0,
        stop_reason=None,
        tool_query=None,
        retrieved_tools=[],
        creator_mandate=None,
        tool_schemas=[],
        executed_tool_calls=[],
        loop_count=0,
        memory_context=[],
        memory_manager=memory_manager,
        subagent_results=[],
        parent_task_id=None,
        task_id=None,
        user_model_context=None,
        shadow_mode=False,
        shadow_log=[],
        # DeepAgent Protocol fields
        input=user_input,
        chat_history=[],
        plan=[],
        past_steps=[],
        scratchpad="",
        criticism="",
        max_loops=max_loops,
        current_loop=0
    )


