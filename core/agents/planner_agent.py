"""
DeepAgent Protocol - Story 2.1: The Hierarchical Planner

Decomposes complex user prompts into atomic steps before execution,
preventing hallucinated actions and ensuring logical consistency.
"""

import logging
from typing import List, Optional, Dict, Any
from core.state import DeepAgentState
from core.llm_api import run_llm
from core.logging import log_event

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Hierarchical planner that decomposes complex prompts into atomic steps.
    
    Story 2.1: Creates numbered action plans that the Executor can follow,
    consulting memory to avoid repeating previously failed approaches.
    """
    
    PLANNER_PROMPT = """You are a strategic planning agent. Your task is to decompose the user's request into a numbered list of atomic, actionable steps.

IMPORTANT RULES:
1. Each step must be specific and actionable
2. Each step should accomplish ONE thing
3. Steps should be in logical order (dependencies first)
4. Use action verbs (Generate, Create, Post, Fetch, Analyze, etc.)
5. If the task is simple (e.g., a greeting), a single step is acceptable

USER REQUEST:
{user_input}

{memory_context}

{failed_plans_context}

Output ONLY a numbered list of steps. No explanations, no preamble.
Format:
1. [First step]
2. [Second step]
...

PLAN:"""

    FAILED_PLANS_PROMPT = """
WARNING: The following similar plans have failed in the past. Learn from these failures:
{failed_plans}

Adjust your plan to avoid these mistakes.
"""

    def __init__(self, memory_manager: Optional[Any] = None):
        """
        Initialize the PlannerAgent.
        
        Args:
            memory_manager: Optional reference to MemoryManager for historical plan lookup
        """
        self.memory_manager = memory_manager
    
    async def plan(self, state: DeepAgentState) -> DeepAgentState:
        """
        Generate a plan from the user's input.
        
        Story 2.1: Decomposes the input into numbered atomic steps,
        consulting memory for similar failed plans.
        
        Args:
            state: Current DeepAgentState with 'input' field populated
            
        Returns:
            Updated state with 'plan' field populated
        """
        user_input = state.get("input", "")
        
        if not user_input:
            log_event("PlannerAgent: No input provided", "WARNING")
            state["plan"] = ["Respond to the user with a helpful message"]
            return state
        
        # Check memory for similar failed plans
        failed_plans_context = await self._get_failed_plans(user_input)
        
        # Build memory context
        memory_context = self._format_memory_context(state.get("memory_context", []))
        
        # Build the prompt
        prompt = self.PLANNER_PROMPT.format(
            user_input=user_input,
            memory_context=memory_context,
            failed_plans_context=failed_plans_context
        )
        
        try:
            result = await run_llm(prompt, purpose="planning")
            plan_text = result.get("result", "")
            
            # Parse the numbered list
            plan_steps = self._parse_plan(plan_text)
            
            if not plan_steps:
                # Fallback to single-step plan
                plan_steps = [f"Respond to: {user_input[:100]}"]
            
            state["plan"] = plan_steps
            log_event(f"PlannerAgent: Generated {len(plan_steps)} step plan", "INFO")
            
            # Log the plan for debugging
            for i, step in enumerate(plan_steps, 1):
                logger.debug(f"  Step {i}: {step}")
            
        except Exception as e:
            log_event(f"PlannerAgent: Failed to generate plan: {e}", "ERROR")
            state["plan"] = [f"Respond to the user's request: {user_input[:100]}"]
        
        return state
    
    def _parse_plan(self, plan_text: str) -> List[str]:
        """
        Parse a numbered list into a list of step strings.
        
        Args:
            plan_text: Raw LLM output containing numbered steps
            
        Returns:
            List of step strings without numbers
        """
        import re
        
        steps = []
        lines = plan_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match patterns like "1. ", "1) ", "Step 1:", etc.
            match = re.match(r'^(?:\d+[\.\)\:]|Step\s*\d+[\.\:])?\s*(.+)$', line, re.IGNORECASE)
            if match:
                step_text = match.group(1).strip()
                if step_text and not step_text.lower().startswith("plan"):
                    steps.append(step_text)
        
        return steps
    
    async def _get_failed_plans(self, user_input: str) -> str:
        """
        Query memory for similar plans that have failed.
        
        Args:
            user_input: Current user query
            
        Returns:
            Formatted string of failed plans, or empty string
        """
        if not self.memory_manager:
            return ""
        
        try:
            # Query for similar failed experiences
            if hasattr(self.memory_manager, "query_wisdom"):
                wisdoms = await self.memory_manager.query_wisdom(
                    query=user_input,
                    filter_source="failure",
                    limit=3
                )
                
                if wisdoms:
                    failed_summaries = []
                    for w in wisdoms:
                        failed_summaries.append(
                            f"- Situation: {w.situation}\n"
                            f"  Action: {w.action}\n"
                            f"  Outcome: {w.outcome}"
                        )
                    
                    return self.FAILED_PLANS_PROMPT.format(
                        failed_plans="\n".join(failed_summaries)
                    )
        except Exception as e:
            logger.warning(f"Failed to query memory for failed plans: {e}")
        
        return ""
    
    def _format_memory_context(self, memory_context: List[Dict[str, Any]]) -> str:
        """
        Format memory context for injection into the planning prompt.
        
        Args:
            memory_context: List of similar past interactions
            
        Returns:
            Formatted context string
        """
        if not memory_context:
            return ""
        
        context_parts = ["RELEVANT CONTEXT FROM PAST INTERACTIONS:"]
        for ctx in memory_context[:3]:  # Limit to top 3
            content = ctx.get("content", "")[:200]
            context_parts.append(f"- {content}")
        
        return "\n".join(context_parts)


async def create_plan(state: DeepAgentState) -> DeepAgentState:
    """
    Graph node function for the Planner.
    
    This is the callable that LangGraph invokes.
    """
    planner = PlannerAgent(memory_manager=state.get("memory_manager"))
    return await planner.plan(state)
