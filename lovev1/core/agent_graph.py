import logging
import asyncio
from typing import TypedDict, Annotated, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from core.dispatcher import global_dispatcher, Event

# We will define the state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    current_step: str
    tools_output: Dict[str, Any]
    final_answer: Union[str, None]
    # Shared mutable context for the event-driven agents
    shared_context: Dict[str, Any]

# --- Agent Logic (Subscribers) ---

async def reason_agent(event: Event):
    """
    Subscribed to TASK_POSTED.
    Analyzes the goal and decides the next step.
    """
    logging.info(f"[ReasonAgent] Received event: {event.type} from {event.source}")
    payload = event.payload
    goal = payload.get("goal")
    
    # In a real implementation, this would call an LLM.
    # For now, we simulate reasoning.
    thought = f"Reasoning about goal: {goal}. I need to take action."
    logging.info(f"[ReasonAgent] {thought}")
    
    # Publish next event
    next_payload = {"goal": goal, "plan": "execute_tools", "context": payload.get("context", {})}
    await global_dispatcher.publish(Event(
        type="ACTION_REQUIRED",
        payload=next_payload,
        source="ReasonAgent"
    ))

async def action_agent(event: Event):
    """
    Subscribed to ACTION_REQUIRED.
    Executes tools.
    """
    logging.info(f"[ActionAgent] Received event: {event.type} from {event.source}")
    payload = event.payload
    
    # Simulate action
    logging.info("[ActionAgent] Executing tools...")
    result = "Tools executed successfully."
    
    # Publish next event
    next_payload = {"goal": payload.get("goal"), "result": result, "context": payload.get("context", {})}
    await global_dispatcher.publish(Event(
        type="REFLECTION_NEEDED",
        payload=next_payload,
        source="ActionAgent"
    ))

async def reflect_agent(event: Event):
    """
    Subscribed to REFLECTION_NEEDED.
    Evaluates the result.
    """
    logging.info(f"[ReflectAgent] Received event: {event.type} from {event.source}")
    payload = event.payload
    result = payload.get("result")
    
    logging.info(f"[ReflectAgent] Reflecting on result: {result}")
    
    # Determine if goal is met
    if "success" in str(result).lower():
         await global_dispatcher.publish(Event(
            type="GOAL_ACHIEVED",
            payload={"final_answer": "Goal Achieved via Event Bus!"},
            source="ReflectAgent"
        ))
    else:
        # Loop back or fail? For demo, we succeed.
        await global_dispatcher.publish(Event(
            type="GOAL_ACHIEVED",
            payload={"final_answer": "Goal assumed achieved."},
            source="ReflectAgent"
        ))

# Subscribe agents to events
global_dispatcher.subscribe("TASK_POSTED", reason_agent)
global_dispatcher.subscribe("ACTION_REQUIRED", action_agent)
global_dispatcher.subscribe("REFLECTION_NEEDED", reflect_agent)

# --- Graph Nodes ---

async def event_loop_node(state: AgentState):
    """
    The main supervisor loop. Processing events from the dispatcher.
    """
    logging.info("Entering Event Loop Node")
    
    # If this is the start and we have a goal, publish the initial event
    if state.get("current_step") == "start" and state.get("goal"):
        logging.info("Publishing initial TASK_POSTED event")
        await global_dispatcher.publish(Event(
            type="TASK_POSTED",
            payload={"goal": state["goal"], "context": state.get("shared_context", {})},
            source="UserEntry"
        ))
        # Update step so we don't repost
        state["current_step"] = "processing"
    
    # Process all pending events
    # This runs the callbacks (agents) which publish new events
    await global_dispatcher.process_events()
    
    # Check for GOAL_ACHIEVED by inspecting a shared mechanism or just relying on
    # the dispatcher to hold the state?
    # Since callbacks are async and returning void, we need a way to look into the "result".
    # For this implementation, we can use a special "System" subscriber that updates a thread-safe
    # object, OR we just check the dispatcher queue?
    # BUT, LangGraph state is local.
    
    # HACK: For the purpose of the graph state update, let's inject a "StateUpdater" agent 
    # that updates a mutable dict passed in the payload, creating a side-effect we can read here?
    # No, keep it clean.
    
    return {"current_step": "cycling"} # Return update to keep loop going

# To properly exit, we need to know if GOAL_ACHIEVED happened.
# Let's add a subscriber that updates a flag we can check.
GOAL_RESULT = {}

async def execution_monitor(event: Event):
    if event.type == "GOAL_ACHIEVED":
        GOAL_RESULT["final_answer"] = event.payload.get("final_answer")

global_dispatcher.subscribe("GOAL_ACHIEVED", execution_monitor)

# Define the graph
workflow = StateGraph(AgentState)

# Add node
workflow.add_node("event_loop", event_loop_node)

# Add edge
workflow.set_entry_point("event_loop")

def should_continue(state: AgentState):
    if "final_answer" in GOAL_RESULT:
        # Inject the final answer into the state for the return
        state["final_answer"] = GOAL_RESULT["final_answer"]
        return END
    return "event_loop"

workflow.add_conditional_edges(
    "event_loop",
    should_continue,
    {
        "event_loop": "event_loop",
        END: END
    }
)

app = workflow.compile()
