from typing import Dict, Any, List
from core.state import DeepAgentState
from core.llm_api import run_llm
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

async def evolution_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Evolution Node:
    1. Analyzes the user's request.
    2. Generates a SMART goal user story.
    3. Calls the 'feed_user_story' tool (conceptually) by returning a tool call message.
    """
    messages = state["messages"]
    last_user_msg = messages[-1].content
    
    # SYSTEM PROMPT
    prompt = """
    You are the Evolution Team Lead.
    Your goal is to take a raw feature request or idea and transform it into a comprehensive, SMART (Specific, Measurable, Achievable, Relevant, Time-bound) user story.
    
    This user story will be fed to 'Jules', a software engineer agent.
    
    Output Format:
    You must call the tool `feed_user_story` with the generated description.
    Do NOT output conversational text. ONLY call the tool.
    
    The description should be standalone and include:
    - Title
    - User Story (As a... I want... So that...)
    - Acceptance Criteria
    - Technical Considerations (if any knowledge available, otherwise keep high level)
    """

    # We use run_llm to get the tool call.
    # Note: run_llm in this codebase might not natively return tool_calls depending on config.
    # If run_llm returns text, we might need to parse it or just use a specific prompt to get JSON.
    # However, `tool_execution_node` expects `tool_calls` in the message.
    
    # Using 'tool_calling' purpose or mode if available, or just standard prompting.
    # Let's assume we can ask for a specific tool call structure or use `function_call` if supported.
    # Since I don't have full visibility into `run_llm`'s tool binding capabilities for *every* model,
    # I will craft a prompt that forces the tool call if the model supports it, 
    # OR I will construct the tool call manually if the output is just the description.
    
    # EASIER PATH: Ask LLM for the SMART *text*, then wrap it in a tool call manually.
    # This is more robust against models that suck at tool calling.
    
    # --- Prevent Duplicates by reading active tasks ---
    import os
    import json
    existing_tasks_context = ""
    try:
        # Assuming love_state.json is in the root directory (two levels up)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        state_path = os.path.join(root_dir, 'love_state.json')
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                love_state = json.load(f)
                
            # Extract tasks from multiple sources to be safe
            active_tasks = []
            
            # 1. Check evolution_history
            if "evolution_history" in love_state:
                for entry in love_state["evolution_history"]:
                     if isinstance(entry, dict) and "task_id" in entry:
                         active_tasks.append(f"- [History] {entry.get('summary', 'Unknown Task')}")
            
            # 2. Check proactive_leads (pending ideas)
            if "proactive_leads" in love_state:
                for lead in love_state["proactive_leads"]:
                    if isinstance(lead, str):
                        active_tasks.append(f"- [Pending] {lead}")
                    elif isinstance(lead, dict):
                         active_tasks.append(f"- [Pending] {lead.get('description', '')}")

            if active_tasks:
                existing_tasks_context = "\nCurrently Active/Pending Tasks (DO NOT DUPLICATE):\n" + "\n".join(active_tasks[-20:]) # Last 20
                
    except Exception as e:
        existing_tasks_context = f"\n(Could not fetch existing tasks: {e})"

    generation_prompt = f"""
    You are the Evolution Team Lead.
    Please convert the following request into a comprehensive SMART user story for a software engineer.
    
    Request: {last_user_msg}
    
    {existing_tasks_context}
    
    INSTRUCTION: 
    1. If the request is a DUPLICATE of an existing task listed above, output "DUPLICATE" and nothing else.
    2. Otherwise, provide ONLY the content of the user story. Do not add intro/outro fluff.
    """
    
    response = await run_llm(generation_prompt, purpose="reasoning")
    smart_story = response.get("result", "").strip()
    
    if smart_story == "DUPLICATE":
        # Handle duplicate gracefully (maybe skip tool call or log it)
        # For now, we return a message indicating no action
        return {"messages": [AIMessage(content="Duplicate task detected. Skipping.")], "next_node": "__end__"}
    
    if not smart_story:
        smart_story = f"Failed to generate story for: {last_user_msg}"
        
    # Construct the tool call message
    # We are simulating that the LLM decided to call this tool.
    import uuid
    call_id = f"call_{str(uuid.uuid4())[:8]}"
    
    tool_call = {
        "name": "feed_user_story",
        "args": {"story": smart_story},
        "id": call_id
    }
    
    # We return an AIMessage with the tool call
    msg = AIMessage(
        content="",
        tool_calls=[tool_call]
    )
    
    return {"messages": [msg], "next_node": "tool_execution_node"}
