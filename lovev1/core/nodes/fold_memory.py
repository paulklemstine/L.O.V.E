"""
Fold Memory Node for the DeepAgent graph.

This node implements Story 2.2: Memory Folding Strategy.
It triggers when the context approaches token limits and creates
"Knowledge Nuggets" that summarize older messages.

The node:
1. Estimates current token usage
2. If > 80% of limit, triggers MemoryFoldingAgent
3. Creates a Knowledge Nugget summary
4. Replaces oldest 50% of messages with the summary
5. Stores the nugget in the knowledge base
"""
import core.logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, SystemMessage


async def fold_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node triggered when reasoning outputs <fold_thought> token or
    when context approaches token limits.
    
    Implements Story 2.2: Memory Folding Strategy.
    
    Args:
        state: The DeepAgentState containing messages and memory_manager
        
    Returns:
        Updated state with folded messages if folding occurred
    """
    messages = state.get("messages", [])
    memory_manager = state.get("memory_manager")
    
    # Configuration
    TOKEN_LIMIT = 4096  # Default context window
    FOLD_THRESHOLD = 0.8  # 80% triggers folding
    
    if not memory_manager:
        core.logging.log_event(
            "Fold memory: No memory_manager in state, skipping.",
            "DEBUG"
        )
        return {}
    
    if not messages or len(messages) < 4:
        # Need at least 4 messages to make folding worthwhile
        return {}
    
    try:
        # Check and fold if necessary
        modified_messages, nugget = await memory_manager.check_and_fold_context(
            messages=messages,
            token_limit=TOKEN_LIMIT,
            threshold=FOLD_THRESHOLD
        )
        
        if nugget:
            core.logging.log_event(
                f"Memory folded: {nugget.source_message_count} messages -> 1 summary, "
                f"saved ~{nugget.token_savings} tokens",
                "INFO"
            )
            
            return {
                "messages": modified_messages,
                "stop_reason": None  # Continue to reasoning after folding
            }
        
        # No folding needed
        return {}
        
    except Exception as e:
        core.logging.log_event(
            f"Fold memory error: {e}",
            "ERROR"
        )
        return {}


def should_trigger_folding(state: Dict[str, Any], token_limit: int = 4096) -> bool:
    """
    Utility function to check if memory folding should be triggered.
    
    Can be used by the graph router to decide whether to route to fold_memory_node.
    
    Args:
        state: The DeepAgentState
        token_limit: Maximum token limit
        
    Returns:
        True if folding should be triggered
    """
    messages = state.get("messages", [])
    memory_manager = state.get("memory_manager")
    
    if not memory_manager or not messages:
        return False
    
    try:
        current_tokens = memory_manager.estimate_token_count(messages)
        threshold_tokens = int(token_limit * 0.8)
        return current_tokens > threshold_tokens
    except Exception:
        return False


def create_folded_message_summary(content: str, message_count: int) -> SystemMessage:
    """
    Creates a formatted system message for the folded summary.
    
    Args:
        content: The summary content
        message_count: Number of messages that were folded
        
    Returns:
        SystemMessage with the formatted summary
    """
    return SystemMessage(
        content=f"""[📦 Memory Folded]
The following is a summary of {message_count} previous messages that have been 
compressed to maintain context efficiency:

---
{content}
---

This summary preserves key directives and context from the earlier conversation."""
    )
