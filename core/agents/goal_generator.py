
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
