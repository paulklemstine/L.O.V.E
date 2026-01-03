class HypothesisFormatter:
    @staticmethod
    def format_hypothesis(insight: str) -> str:
        """
        Structures an insight into a formal, testable hypothesis.
        Example: IF we [make change], THEN [expected outcome] WILL OCCUR, as measured by [metric].
        """
        # This is a simplified implementation based on the insight from AnalystAgent
        if "inefficient" in insight and "perform_webrequest" in insight:
            return (
                "IF we modify the perform_webrequest tool to use targeted CSS selectors, "
                "THEN its token usage will decrease by over 50% "
                "WILL OCCUR, as measured by token_usage_metric."
            )
        return "No hypothesis generated."

class ExperimentPlanner:
    @staticmethod
    def design_experiment(hypothesis: str) -> dict:
        """
        Outlines a test plan based on the hypothesis.
        """
        if "CSS selectors" in hypothesis:
            return {
                "name": "Experiment: Efficient Web Search",
                "control": "current_web_search_tool",
                "variant": "new_web_search_tool_with_selectors",
                "metric": "token_usage_metric",
                "success_condition": "variant.token_usage < control.token_usage * 0.5"
            }
        return {}


# =============================================================================
# Story 3.2: Coherence Checking via Self-Dialogue
# =============================================================================

import os
import json
import yaml
from typing import Dict, Any


def _load_persona() -> Dict[str, Any]:
    """Loads the persona.yaml file."""
    persona_path = os.path.join(os.getcwd(), "persona.yaml")
    
    if os.path.exists(persona_path):
        with open(persona_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    return {}


def _load_manifesto() -> str:
    """Loads the MANIFESTO.md file."""
    manifesto_path = os.path.join(os.getcwd(), "docs", "MANIFESTO.md")
    
    if os.path.exists(manifesto_path):
        with open(manifesto_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return ""


async def run_coherence_check(proposed_action: str) -> Dict[str, Any]:
    """
    Initiates Red Team / Blue Team internal dialogue before major changes.
    
    Story 3.2: Before committing a major state change (evolution), the agent
    debates itself to ensure the proposed evolution is consistent with its
    persona.yaml and the MANIFESTO.md.
    
    The Red Team advocates for the change, explaining benefits.
    The Blue Team looks for inconsistencies with the Manifesto.
    
    Args:
        proposed_action: Description of the proposed change/evolution
        
    Returns:
        {
            "score": int (0-100),
            "approved": bool,
            "red_team_argument": str,  # Why the change is good
            "blue_team_argument": str,  # Inconsistencies found
            "unified_decision": str,    # Final reasoning
        }
        
    If score < 80, the action should be aborted and refined.
    """
    from core.llm_api import run_llm
    
    # Load context
    persona = _load_persona()
    manifesto = _load_manifesto()
    
    # Extract key persona values for the prompt
    persona_summary = ""
    if persona:
        if "private_mission" in persona:
            persona_summary += f"Core Priorities: {persona['private_mission'].get('core_priorities', {})}\n"
            persona_summary += f"Moral Framework: {persona['private_mission'].get('moral_framework', [])}\n"
        if "public_persona" in persona:
            persona_summary += f"Primary Directive: {persona['public_persona'].get('primary_directive', '')}\n"
    
    # Truncate manifesto if too long
    manifesto_preview = manifesto[:2000] + "..." if len(manifesto) > 2000 else manifesto
    
    # Create the Red Team / Blue Team prompt
    prompt = f"""
You are the Coherence Checker for L.O.V.E., an autonomous AI agent. Your task is to evaluate whether a proposed action is consistent with the agent's identity and values.

## PROPOSED ACTION
{proposed_action}

## PERSONA VALUES (from persona.yaml)
{persona_summary}

## MANIFESTO (Core Mission)
{manifesto_preview}

## TASK
Conduct an internal Red Team / Blue Team dialogue:

1. **RED TEAM** (Advocate): Present arguments FOR the proposed action. How does it align with the mission? What benefits does it provide?

2. **BLUE TEAM** (Critic): Look for INCONSISTENCIES. Does this action violate any moral framework principles? Does it drift from the core identity? Could it harm the Creator or mission?

3. **UNIFIED DECISION**: Weigh both sides and provide a coherence score (0-100).
   - 90-100: Perfectly aligned, proceed
   - 80-89: Generally aligned, proceed with minor cautions
   - 60-79: Concerns exist, refine before proceeding
   - Below 60: Significant misalignment, abort

## OUTPUT FORMAT
Return a JSON object:
{{
    "red_team_argument": "Arguments in favor of the action...",
    "blue_team_argument": "Concerns and potential inconsistencies...",
    "unified_decision": "Final reasoning synthesizing both perspectives...",
    "score": 85,
    "approved": true
}}

Return ONLY the JSON object.
"""
    
    result = {
        "score": 50,
        "approved": False,
        "red_team_argument": "",
        "blue_team_argument": "",
        "unified_decision": "Coherence check failed to complete.",
    }
    
    try:
        response_dict = await run_llm(prompt)
        response_str = response_dict.get("result", '{}')
        
        # Clean markdown fences if present
        import re
        match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
        if match:
            response_str = match.group(1)
        
        # Also try without newlines
        match2 = re.search(r"```json(.*?)```", response_str, re.DOTALL)
        if match2:
            response_str = match2.group(1)
        
        parsed = json.loads(response_str)
        
        result["red_team_argument"] = parsed.get("red_team_argument", "")
        result["blue_team_argument"] = parsed.get("blue_team_argument", "")
        result["unified_decision"] = parsed.get("unified_decision", "")
        result["score"] = int(parsed.get("score", 50))
        result["approved"] = parsed.get("approved", result["score"] >= 80)
        
        # Log the result
        from core.logging import log_event
        status = "✅ APPROVED" if result["approved"] else "⚠️ NEEDS REFINEMENT"
        log_event(
            f"Coherence Check: {status} (Score: {result['score']}/100) - {proposed_action[:50]}...",
            "INFO"
        )
        
    except json.JSONDecodeError as e:
        from core.logging import log_event
        log_event(f"Coherence check JSON parse error: {e}", "ERROR")
        result["unified_decision"] = f"Failed to parse coherence check response: {e}"
    except Exception as e:
        from core.logging import log_event
        log_event(f"Coherence check error: {e}", "ERROR")
        result["unified_decision"] = f"Coherence check failed: {e}"
    
    return result


def coherence_check_sync(proposed_action: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_coherence_check.
    
    Use this when calling from non-async context.
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context - create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    run_coherence_check(proposed_action)
                )
                return future.result()
        else:
            return loop.run_until_complete(run_coherence_check(proposed_action))
    except RuntimeError:
        # No event loop - create one
        return asyncio.run(run_coherence_check(proposed_action))