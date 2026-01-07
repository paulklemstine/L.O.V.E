"""
DeepAgent Protocol - Story 2.2: The Self-Reflective Critic

Analyzes execution results against original intent to enable self-correction.
If unsatisfactory, updates scratchpad with specific correction instructions.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from core.state import DeepAgentState
from core.llm_api import run_llm
from core.logging import log_event

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Self-reflective critic that validates execution quality.
    
    Story 2.2: Analyzes past_steps vs original input to determine
    if the task was completed satisfactorily. Provides correction
    instructions for the next iteration if needed.
    """
    
    CRITIC_PROMPT = """You are a quality assurance critic. Analyze whether the executed steps successfully addressed the user's original request.

ORIGINAL REQUEST:
{original_input}

PLANNED STEPS:
{plan}

EXECUTED STEPS AND RESULTS:
{past_steps}

CURRENT SCRATCHPAD (previous corrections if any):
{scratchpad}

Evaluate the execution:
1. Did the steps address the user's request?
2. Were there any errors or failures?
3. Is the output quality satisfactory?
4. Should we try again with a different approach?

Respond with ONLY valid JSON in this exact format:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "feedback": "Brief explanation of your judgment",
    "corrections": "If not approved, specific instructions for the next attempt. If approved, leave empty."
}}

YOUR JUDGMENT:"""

    def __init__(self, approval_threshold: float = 0.7):
        """
        Initialize the CriticAgent.
        
        Args:
            approval_threshold: Minimum confidence to auto-approve (0.0-1.0)
        """
        self.approval_threshold = approval_threshold
    
    async def critique(self, state: DeepAgentState) -> DeepAgentState:
        """
        Analyze execution results and provide feedback.
        
        Story 2.2: Compares past_steps against original input,
        updating scratchpad with corrections if needed.
        
        Args:
            state: Current DeepAgentState with past_steps populated
            
        Returns:
            Updated state with criticism and potentially updated scratchpad
        """
        original_input = state.get("input", "")
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])
        scratchpad = state.get("scratchpad", "")
        current_loop = state.get("current_loop", 0)
        max_loops = state.get("max_loops", 5)
        
        # Format past steps for analysis
        formatted_steps = self._format_past_steps(past_steps)
        formatted_plan = self._format_plan(plan)
        
        # Build the prompt
        prompt = self.CRITIC_PROMPT.format(
            original_input=original_input,
            plan=formatted_plan,
            past_steps=formatted_steps,
            scratchpad=scratchpad if scratchpad else "(No previous corrections)"
        )
        
        try:
            result = await run_llm(prompt, purpose="critique")
            critique_text = result.get("result", "")
            
            # Parse the JSON response
            judgment = self._parse_judgment(critique_text)
            
            approved = judgment.get("approved", False)
            confidence = judgment.get("confidence", 0.5)
            feedback = judgment.get("feedback", "")
            corrections = judgment.get("corrections", "")
            
            # Apply approval threshold
            if confidence >= self.approval_threshold and approved:
                state["criticism"] = f"APPROVED (confidence: {confidence:.2f}): {feedback}"
                state["next_node"] = "finalize"
                log_event(f"CriticAgent: APPROVED - {feedback[:100]}", "INFO")
            else:
                # Check if we've exceeded max loops
                if current_loop >= max_loops - 1:
                    state["criticism"] = f"MAX_LOOPS_REACHED: {feedback}"
                    state["next_node"] = "finalize"
                    state["stop_reason"] = f"Max loops ({max_loops}) reached. Last feedback: {feedback}"
                    log_event(f"CriticAgent: Max loops reached, finalizing", "WARNING")
                else:
                    # Update scratchpad with corrections
                    new_scratchpad = self._update_scratchpad(scratchpad, corrections, current_loop)
                    state["scratchpad"] = new_scratchpad
                    state["criticism"] = f"REJECTED (confidence: {confidence:.2f}): {feedback}"
                    state["next_node"] = "planner"
                    state["current_loop"] = current_loop + 1
                    log_event(f"CriticAgent: REJECTED - Loop {current_loop + 1}/{max_loops}", "INFO")
            
        except Exception as e:
            log_event(f"CriticAgent: Failed to critique: {e}", "ERROR")
            # Default to approved to prevent infinite loops on error
            state["criticism"] = f"ERROR during critique: {e}"
            state["next_node"] = "finalize"
        
        return state
    
    def _format_past_steps(self, past_steps: list) -> str:
        """Format past steps for the prompt."""
        if not past_steps:
            return "(No steps executed yet)"
        
        formatted = []
        for i, step in enumerate(past_steps, 1):
            if isinstance(step, tuple) and len(step) >= 3:
                desc, action, result = step[0], step[1], step[2]
                formatted.append(f"{i}. Step: {desc}\n   Action: {action}\n   Result: {result}")
            else:
                formatted.append(f"{i}. {step}")
        
        return "\n".join(formatted)
    
    def _format_plan(self, plan: list) -> str:
        """Format the plan for the prompt."""
        if not plan:
            return "(No plan available)"
        
        return "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))
    
    def _parse_judgment(self, critique_text: str) -> Dict[str, Any]:
        """
        Parse the critic's JSON response.
        
        Args:
            critique_text: Raw LLM output
            
        Returns:
            Parsed judgment dict with approved, confidence, feedback, corrections
        """
        # Try to extract JSON from the response
        try:
            # Clean up common issues
            text = critique_text.strip()
            
            # Find JSON boundaries
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to infer from text
        text_lower = critique_text.lower()
        
        approved = any(word in text_lower for word in ["approved", "success", "complete", "satisfactory"])
        rejected = any(word in text_lower for word in ["rejected", "failed", "error", "unsatisfactory", "try again"])
        
        return {
            "approved": approved and not rejected,
            "confidence": 0.5,
            "feedback": critique_text[:200],
            "corrections": "" if approved else "Please review and try a different approach."
        }
    
    def _update_scratchpad(self, current: str, corrections: str, loop: int) -> str:
        """
        Update the scratchpad with new corrections.
        
        Args:
            current: Current scratchpad content
            corrections: New corrections to add
            loop: Current loop number
            
        Returns:
            Updated scratchpad
        """
        if not corrections:
            return current
        
        new_entry = f"\n--- Loop {loop + 1} Corrections ---\n{corrections}\n"
        
        # Keep scratchpad from growing too large
        max_length = 2000
        combined = current + new_entry
        
        if len(combined) > max_length:
            # Keep the most recent corrections
            combined = combined[-max_length:]
            # Find the start of a complete entry
            start = combined.find("--- Loop")
            if start > 0:
                combined = combined[start:]
        
        return combined


async def critique_execution(state: DeepAgentState) -> DeepAgentState:
    """
    Graph node function for the Critic.
    
    This is the callable that LangGraph invokes.
    """
    critic = CriticAgent()
    return await critic.critique(state)


def decide_next_node(state: DeepAgentState) -> str:
    """
    Conditional edge function for the graph.
    
    Returns the name of the next node based on the critic's decision.
    """
    return state.get("next_node", "finalize")
