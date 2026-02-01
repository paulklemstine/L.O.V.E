"""
Tool Gap Detector - Evolutionary Awareness

Epic 1, Story 1.1: Identifies when the agent lacks necessary tools and 
generates specifications for new ones.

Functionality:
1. Listens to ToolRetriever for low-confidence searches.
2. Analyzes the gap using LLM.
3. Creates EvolutionarySpecification in evolution_state.
"""

import logging
import json
import re
import asyncio
import threading
from typing import Optional, Dict, Any

from core.logger import log_event
from core.evolution_state import (
    EvolutionarySpecification,
    add_tool_specification,
    get_pending_specifications
)
from core.tool_retriever import get_tool_retriever

logger = logging.getLogger(__name__)


class ToolGapDetector:
    """
    Detects missing capabilities and generates specifications for new tools.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.retriever = get_tool_retriever()
        
        from core.feature_flags import ENABLE_TOOL_EVOLUTION
        
        if ENABLE_TOOL_EVOLUTION:
            # Subscribe to retriever events
            self.retriever.add_gap_listener(self._on_gap_detected)
            log_event("ToolGapDetector initialized and listening", "INFO")
        else:
            log_event("ToolGapDetector disabled by feature flag", "INFO")
        
    def _get_llm_client(self):
        if self.llm_client is None:
            from core.llm_client import get_llm_client
            self.llm_client = get_llm_client()
        return self.llm_client

    def _on_gap_detected(self, step_description: str, best_score: float):
        """Callback when retrieval fails to find a good match."""
        # Story 1.1: Only trigger if we don't already have a pending spec for this
        # Simple deduplication based on recent pending specs
        pending = get_pending_specifications()
        for spec in pending:
            if spec.trigger_context == step_description:
                # Already tracking this gap
                return
        
        log_event(
            f"🔍 Tool Gap Detected! (Score: {best_score:.2f}) for: '{step_description}'", 
            "WARNING"
        )
        
        # We don't block the retrieval (async handling would be ideal here)
        # For now, we just log it and maybe generate spec immediately 
        # depending on architecture.
        # In this synchronous implementation, we might want to defer generation 
        # to a background task or just do it if it's critical.
        
        # For the prototype, we'll generate it now but acknowledge latency
        try:
            # Run analysis in a separate thread to avoid blocking the main loop
            def run_analysis():
                try:
                    asyncio.run(self.analyze_gap_and_specify(step_description))
                except Exception as e:
                    logger.error(f"Background gap analysis failed: {e}")

            threading.Thread(target=run_analysis, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to schedule gap analysis: {e}")

    async def analyze_gap_and_specify(self, context: str) -> Optional[EvolutionarySpecification]:
        """
        Analyze the missing capability and generate a specification.
        """
        log_event(f"🧬 Analyzing evolutionary gap for: {context}", "INFO")
        
        prompt = f"""
You are the Evolutionary Architect for L.O.V.E.
The agent encountered a step for which no suitable tool was found.

CONTEXT: "{context}"

Your task is to define a NEW tool that would solve the underlying *primitive* problem.

CRITICAL:
- Do NOT create a tool that just solves this specific context (e.g., "create_posters_for_investors").
- Create an ATOMIC, REUSABLE tool (e.g., "generate_image", "overlay_text").
- The tool must be S.M.A.R.T.: Specific, Measurable, Achievable, Relevant, Time-bound.

EXAMPLES:
BAD (Too specific): `generate_demotivational_posters_for_campaign`
GOOD (Atomic): `generate_image_from_prompt`

BAD (Too complex): `search_twitter_and_reply_to_influencers`
GOOD (Atomic): `search_twitter_users`, `post_reply`

REQUIREMENTS:
1. Determine the function name (snake_case, verb_noun).
2. Define required arguments (name: type).
3. Define expected output.
4. List safety constraints.
5. Provide reasoning for why this is the right atomic primitive.
6. Return JSON ONLY.

FORMAT:
{{
    "functional_name": "verb_noun",
    "required_arguments": {{
        "prompt": "str",
        "count": "int"
    }},
    "expected_output": "description of return value",
    "safety_constraints": [
        "must not delete files",
        "must handle errors"
    ],
    "reasoning": "This tool provides the primitive capability to X, which allows solving the context by..."
}}
"""
        llm = self._get_llm_client()
        # Use async generation with low temperature for architecture reasoning
        response = await llm.generate_async(prompt, temperature=0.2)
        
        # Parse JSON
        try:
            # Clean markdown
            if "```" in response:
                response = re.sub(r'```json\n|\n```|```', '', response)
            
            data = json.loads(response.strip())
            
            spec = EvolutionarySpecification(
                functional_name=data.get("functional_name", "unknown_tool"),
                required_arguments=data.get("required_arguments", {}),
                expected_output=data.get("expected_output", ""),
                safety_constraints=data.get("safety_constraints", []),
                trigger_context=context,
                status="pending"
            )
            
            # Save to state
            add_tool_specification(spec)
            
            log_event(
                f"✅ Created evolutionary spec: {spec.functional_name} (ID: {spec.id})", 
                "INFO"
            )
            return spec
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response for gap analysis: {response}")
            return None
        except Exception as e:
            logger.error(f"Error creating specification: {e}")
            return None

# Singleton
_tool_gap_detector = None

def get_gap_detector():
    global _tool_gap_detector
    if _tool_gap_detector is None:
        _tool_gap_detector = ToolGapDetector()
    return _tool_gap_detector
