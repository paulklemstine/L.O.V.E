"""
Memory Bridge Node for the DeepAgent graph.

This node implements Story 2.1: The Semantic Memory Bridge.
It queries the FAISS vector database for similar past user stories or code 
patches before reasoning begins, preventing repeated mistakes and duplicate effort.

The node:
1. Extracts the user request from the most recent message
2. Performs semantic search on the memory system
3. Injects top-3 relevant past interactions into the state
4. Updates state with memory context for the reasoning node
"""
import core.logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage

# Import will be done dynamically to avoid circular imports
# from core.memory.memory_manager import MemoryManager


async def memory_bridge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queries FAISS for similar past interactions before reasoning begins.
    
    This node is part of the Semantic Memory Bridge (Story 2.1).
    
    Args:
        state: The DeepAgentState containing messages and other context
        
    Returns:
        Updated state with memory_context field populated
    """
    messages = state.get("messages", [])
    memory_manager = state.get("memory_manager")
    
    # Default empty context if no memory manager available
    if not memory_manager:
        core.logging.log_event(
            "Memory bridge: No memory_manager in state, skipping semantic search.",
            "DEBUG"
        )
        return {"memory_context": []}
    
    # Extract the user request from the most recent HumanMessage
    user_request = _extract_user_request(messages)
    
    if not user_request:
        core.logging.log_event(
            "Memory bridge: No user request found in messages.",
            "DEBUG"
        )
        return {"memory_context": []}
    
    try:
        # Perform semantic search for similar past interactions
        similar_interactions = await memory_manager.search_similar_interactions(
            query=user_request,
            top_k=3
        )
        
        if similar_interactions:
            core.logging.log_event(
                f"Memory bridge: Found {len(similar_interactions)} relevant past interactions.",
                "INFO"
            )
            
            # Log the found memories for debugging
            for i, interaction in enumerate(similar_interactions):
                tags = interaction.get("tags", [])
                core.logging.log_event(
                    f"  [{i+1}] Tags: {tags}, Score: {interaction.get('similarity_score', 'N/A'):.4f}",
                    "DEBUG"
                )
        else:
            core.logging.log_event(
                "Memory bridge: No relevant past interactions found.",
                "DEBUG"
            )
        
        return {"memory_context": similar_interactions}
        
    except Exception as e:
        core.logging.log_event(
            f"Memory bridge error: {e}",
            "ERROR"
        )
        return {"memory_context": []}


def _extract_user_request(messages: List[BaseMessage]) -> str:
    """
    Extracts the user request from the message list.
    
    Looks for the most recent HumanMessage and extracts its content.
    If the content is complex (dict), attempts to extract meaningful text.
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        The user request string, or empty string if not found
    """
    # Iterate in reverse to find the most recent HumanMessage
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            
            # Handle string content directly
            if isinstance(content, str):
                return content.strip()
            
            # Handle dict content (e.g., multimodal messages)
            if isinstance(content, dict):
                # Try common keys
                for key in ["text", "content", "query", "request", "message"]:
                    if key in content and isinstance(content[key], str):
                        return content[key].strip()
                # Fallback: stringify the dict
                return str(content)
            
            # Handle list content (e.g., content blocks)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                return " ".join(text_parts).strip()
    
    return ""


def format_memory_context_for_prompt(memory_context: List[Dict[str, Any]]) -> str:
    """
    Formats the memory context for injection into the reasoning prompt.
    
    This is a standalone function that can be used by the reasoning node
    to format memory context without needing the MemoryManager instance.
    
    Args:
        memory_context: List of similar interaction dicts from memory_bridge_node
        
    Returns:
        Formatted markdown string for prompt injection
    """
    if not memory_context:
        return ""
    
    lines = [
        "## 🧠 Relevant Past Interactions",
        "",
        "Before proceeding, consider these similar past interactions:",
        ""
    ]
    
    for i, interaction in enumerate(memory_context, 1):
        lines.append(f"### Past Interaction {i}")
        
        # Add contextual description if available
        desc = interaction.get("contextual_description")
        if desc:
            lines.append(f"**Summary:** {desc}")
        
        # Add truncated content
        content = interaction.get("content", "")
        lines.append(f"**Context:** {content}")
        
        # Add tags if available
        tags = interaction.get("tags", [])
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        
        # Add keywords if available
        keywords = interaction.get("keywords", [])
        if keywords:
            lines.append(f"**Keywords:** {', '.join(keywords)}")
        
        lines.append("")  # Blank line separator
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)
