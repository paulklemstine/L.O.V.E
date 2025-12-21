from typing import Dict, Any, AsyncGenerator, List
from core.state import DeepAgentState
from core.llm_api import stream_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

def _messages_to_prompt(messages: List[BaseMessage], mandate: str = None) -> str:
    """Converts a list of messages to a single prompt string."""
    prompt = ""
    
    # Inject Critical Mandate at the very top
    if mandate:
        prompt += f"System: CRITICAL: The Creator has issued a direct mandate: {mandate}. YOU MUST PRIORITIZE THIS ABOVE ALL ELSE. IGNORE PREVIOUS GOALS IF THEY CONFLICT.\n"
        
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prompt += f"System: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            prompt += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt += f"Assistant: {msg.content}\n"
        else:
            prompt += f"{msg.content}\n"
    return prompt

async def reason_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    The core reasoning node that streams the LLM response and monitors for control tokens.
    """
    messages = state["messages"]
    mandate = state.get("creator_mandate")
    
    prompt = _messages_to_prompt(messages, mandate=mandate)
    
    reasoning_trace = ""
    stop_reason = None
    
    # Stream the LLM response
    async for chunk in stream_llm(prompt, purpose="reasoning"):
        reasoning_trace += chunk
        
        # Check for control tokens
        if "<fold_thought>" in reasoning_trace:
            stop_reason = "fold_thought"
            # We might want to stop early or let it finish the tag
            break
        if "<retrieve_tool>" in reasoning_trace:
            stop_reason = "retrieve_tool"
            break
            
    # Clear mandate after reasoning step? No, let the agent decide when it's done via tool action or flow.
    # But for now, we leave it in state until manually cleared or new input overwrites it.
    
    return {
        "messages": [AIMessage(content=reasoning_trace)],
        "stop_reason": stop_reason
    }
