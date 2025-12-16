from typing import Dict, Any, Literal
from core.state import DeepAgentState
from core.llm_api import run_llm
from langchain_core.messages import SystemMessage, HumanMessage

async def supervisor_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Supervisor node that decides which subgraph or node to route to.
    """
    messages = state["messages"]
    
    # Construct a prompt for the supervisor
    prompt = """
    You are the Supervisor of the DeepAgent system.
    Your goal is to route the current task to the most appropriate team or node.
    
    Available options:
    - "research_team": For tasks requiring extensive information gathering, web search, or analysis.
    - "coding_team": For tasks involving writing, testing, or fixing code.
    - "evolution_team": For tasks related to self-improvement, system updates, or architectural changes.
    - "social_media_team": For managing the social media presence, posting to Bluesky, storytelling, or replying to fans.
    - "reasoning_node": For general reasoning, simple tasks, or when unsure.
    
    Analyze the conversation history and determine the next step.
    
    CRITICAL: You MUST respond with a JSON object wrapped in <json> tags.
    Format:
    <json>
    {"next_node": "coding_team"}
    </json>
    """
    
    # We might want to pass the last few messages to context
    last_msg = messages[-1].content if messages else ""
    prompt += f"\nLast User Message: {last_msg}\n"
    
    response = await run_llm(prompt, purpose="supervisor")
    
    # Use the robust parser
    from core.llm_parser import smart_parse_llm_response
    parsed = smart_parse_llm_response(response.get("result", ""))
    
    decision = parsed.get("next_node", "").strip().lower()
    
    # Validate decision
    valid_options = ["research_team", "coding_team", "evolution_team", "social_media_team", "reasoning_node"]
    
    if decision not in valid_options:
        # Fallback logic
        print(f"[Supervisor] Invalid decision '{decision}', defaulting to reasoning_node")
        decision = "reasoning_node"
        
    return {"next_node": decision}