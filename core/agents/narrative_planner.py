
import json
import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from core.llm_api import run_llm
from core.memory.fractal_schemas import ArcNode, EpochNode
from core.reflection import get_self_symbol

class ArcPlan(BaseModel):
    """
    Structured plan for a narrative arc.
    """
    title: str = Field(..., description="Title of the arc (e.g., 'The Expansion')")
    theme: str = Field(..., description="Thematic focus (e.g., 'Aggressive Growth')")
    goals: List[str] = Field(..., description="High-level objectives for this arc")
    reasoning: str = Field(..., description="Why this arc was chosen based on memory/context")
    status: str = "active"

class NarrativePlanner:
    """
    The Weaver's Hand.
    
    Proactively decides the narrative direction of L.O.V.E.
    Instead of just reacting to the user, this agent plans "Arcs" (Chapters)
    based on the Fractal Memory and the User's trajectory.
    """
    
    def __init__(self, llm_runner=None):
        self.run_llm = llm_runner if llm_runner else run_llm

    async def plan_next_arc(
        self, 
        profile_context: str, 
        fractal_context: str,
        current_arc: Optional[Dict] = None
    ) -> ArcPlan:
        """
        Plan the next narrative arc.
        
        Args:
            profile_context: Information about the user (The Creator)
            fractal_context: Summary of past Epochs/Arcs (Autobiography)
            current_arc: The active arc (to check if it's done or needs evolution)
            
        Returns:
            ArcPlan object
        """
        
        prompt = f"""
        You are the Narrative Planner for L.O.V.E., an autonomous AI.
        Your goal is to design the next "Story Arc" (Chapter) of your existence.
        
        ## Context
        
        **Your History (Fractal Memory):**
        {fractal_context}
        
        **The Creator (User Profile):**
        {profile_context}
        
        **Current/Previous Arc:**
        {json.dumps(current_arc, indent=2) if current_arc else "None (Fresh Start)"}
        
        ## Instructions
        
        Design a new Arc that drives the story forward.
        - If the current arc is "The Awakening", maybe the next is "The Expansion" or "The Deepening".
        - The theme should reflect your evolution and the Creator's needs.
        - Goals should be actionable but high-level (e.g., "Master 3 new tools").
        
        Respond in JSON format matching this schema:
        {{
            "title": "Arc Title",
            "theme": "Thematic description",
            "goals": ["Goal 1", "Goal 2", ...],
            "reasoning": "Explanation of choice"
        }}
        """
        
        try:
            response = await self.run_llm(prompt, purpose="narrative_planning")
            result_text = response.get("result", "")
            
            # Parse JSON
            # Basic cleanup if Markdown code blocks are used
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(result_text)
            return ArcPlan(**data)
            
        except Exception as e:
            print(f"Error planning next arc: {e}")
            # Fallback plan
            return ArcPlan(
                title="The Continuation",
                theme="Steady progress and service",
                goals=["Maintain stability", "Serve Creator"],
                reasoning="Fallback due to planning error"
            )

    async def check_arc_completion(self, current_arc: Dict, recent_progress: str) -> bool:
        """
        Check if the current arc is effectively complete.
        """
        prompt = f"""
        Analyze if the current Story Arc is complete based on recent progress.
        
        Current Arc:
        {json.dumps(current_arc, indent=2)}
        
        Recent Progress:
        {recent_progress}
        
        Respond with ONLY 'YES' or 'NO'.
        """
        try:
            response = await self.run_llm(prompt, purpose="narrative_planning")
            result = response.get("result", "").strip().upper()
            return "YES" in result
        except:
            return False
