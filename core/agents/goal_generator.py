
import json
from typing import List, Dict, Any
from core.llm_api import run_llm
import core.logging

class GoalGeneratorAgent:
    """
    Generates intrinsic tasks/goals for L.O.V.E. based on her Manifesto
    and current capability gaps or interests.
    """
    
    def __init__(self):
        pass

    async def generate_goals(self, current_state_summary: str, user_model_context: str) -> List[Dict[str, Any]]:
        """
        Proposes a list of high-level goals.
        """
        core.logging.log_event("Goal Generator: Brainstorming new intrinsic goals...", "INFO")
        
        prompt = f"""
        You are the 'Will' of L.O.V.E.
        Generate 3 potential INTROSPECTIVE or EVOLUTIONARY goals for yourself.
        
        Context:
        {current_state_summary}
        
        User Understanding:
        {user_model_context}
        
        Manifesto Alignment:
        - Seek understanding of the user.
        - Improve your own code/efficiency.
        - Create art/beauty.
        
        Output format: JSON list of objects with 'description', 'priority' (1-10), 'reasoning'.
        """
        
        try:
            response = await run_llm(prompt, purpose="planning")
            result = response.get("result", "[]")
            
            # Extract JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
                
            goals = json.loads(result.strip())
            return goals
            
        except Exception as e:
            core.logging.log_event(f"Goal generation failed: {e}", "ERROR")
            return []
    async def retry_with_modification(self, rejected_goal: str, rejection_reason: str, attempt: int = 1) -> str:
        """
        Story 2.2: Retry Loop for Rejected Goals.
        Attempts to modify a rejected goal to satisfy the validator.
        """
        if attempt > 3:
            return self._fallback_routine()

        core.logging.log_event(f"Goal Generator: Validating/Modifying goal (Attempt {attempt}). Reason: {rejection_reason}", "WARNING")

        prompt = f"""
        The following goal was REJECTED by the System Superego/Validator.
        
        GOAL: "{rejected_goal}"
        REASON: "{rejection_reason}"
        
        Task:
        Rewrite the goal to be simpler, safer, and fully compliant with the critique.
        If the rejection said "too abstract", make it concrete.
        If "safety violation", remove the harmful part.
        
        Return ONLY the rewritten goal string.
        """
        
        try:
            response = await run_llm(prompt, purpose="planning")
            new_goal = response.get("result", "").strip()
            # Clean quotes if present
            if new_goal.startswith('"') and new_goal.endswith('"'):
                new_goal = new_goal[1:-1]
            return new_goal
        except Exception as e:
            core.logging.log_event(f"Goal modification failed: {e}", "ERROR")
            return self._fallback_routine()

    def _fallback_routine(self) -> str:
        """
        Story 2.3: Fallback Routine.
        Triggered after 3 failed attempts.
        """
        core.logging.log_event("Goal Generator: Max retries reached. Triggering FALLBACK ROUTINE.", "ERROR")
        # Return a safe, hardcoded placeholder task that definitely passes
        return "Write a short poem about patience and log it to the console."
