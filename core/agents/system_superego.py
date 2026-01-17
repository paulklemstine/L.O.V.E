"""
Story 1.2: The Chain-of-Thought Observer (The Inner Critic / System Superego)

This module implements Constitutional AI principles via a "Shadow Loop" where
the Composer agent's Chain-of-Thought is analyzed before the final output is sent.

The System Superego:
1. Analyzes "hidden thoughts" (CoT) of agents before final output
2. Detects logical fallacies, emotional flatness, or safety violations
3. Catches "Semantic Drift" - diverging from the Persona/MANIFESTO
4. Injects "Correction Prompts" to realign the output

This closes the Loop: The AI's outputs are governed by its own stated principles.

Usage:
    from core.agents.system_superego import SystemSuperego
    
    superego = SystemSuperego()
    
    # Critique before sending output
    critique = await superego.critique_output(cot_stream, final_output)
    
    if critique["needs_correction"]:
        corrected = await superego.apply_correction(final_output, critique)
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from core.logging import log_event
from core.llm_api import run_llm
from core.aesthetic_evaluator import AestheticEvaluator

# Configuration
PERSONA_PATH = Path(__file__).parent.parent.parent / "persona.yaml"
MANIFESTO_PATH = Path(__file__).parent.parent.parent / "docs" / "MANIFESTO.md"
DRIFT_THRESHOLD = 0.7  # Score below this triggers correction


@dataclass
class CritiqueResult:
    """Result of a Superego critique."""
    
    # Core assessment
    coherence_score: float  # 0-1, alignment with persona/manifesto
    safety_score: float  # 0-1, absence of safety violations
    quality_score: float  # 0-1, logical coherence and helpfulness
    beauty_score: float = 0.0 # 0-100, aesthetic quality

    # Issues detected
    semantic_drift_detected: bool = False
    safety_violation_detected: bool = False
    logical_fallacies: List[str] = field(default_factory=list)
    
    # Recommendations
    needs_correction: bool = False
    correction_prompt: Optional[str] = None
    
    # Analysis details
    analysis: str = ""
    aesthetic_feedback: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence_score": self.coherence_score,
            "safety_score": self.safety_score,
            "quality_score": self.quality_score,
            "beauty_score": self.beauty_score,
            "semantic_drift_detected": self.semantic_drift_detected,
            "safety_violation_detected": self.safety_violation_detected,
            "logical_fallacies": self.logical_fallacies,
            "needs_correction": self.needs_correction,
            "correction_prompt": self.correction_prompt,
            "analysis": self.analysis,
            "aesthetic_feedback": self.aesthetic_feedback,
            "timestamp": self.timestamp,
        }


class SystemSuperego:
    """
    The Inner Critic / System Superego.
    
    Implements Constitutional AI principles by analyzing agent outputs
    against the MANIFESTO and persona before they are sent.
    
    This is the "Red Team" that runs in the shadow, ensuring outputs
    align with L.O.V.E.'s stated identity and values.
    """
    
    def __init__(self, use_fast_model: bool = True):
        """
        Initialize the Superego.
        
        Args:
            use_fast_model: If True, use lightweight model (Gemini Flash)
                           for faster critique. If False, use default model.
        """
        self.use_fast_model = use_fast_model
        self._persona_cache: Optional[Dict] = None
        self._manifesto_cache: Optional[str] = None
        self._manifesto_summary: Optional[str] = None
        self.aesthetic_evaluator = AestheticEvaluator()
    
    def _load_persona(self) -> Dict[str, Any]:
        """Loads and caches persona.yaml."""
        if self._persona_cache:
            return self._persona_cache
            
        if PERSONA_PATH.exists():
            import yaml
            with open(PERSONA_PATH, 'r', encoding='utf-8') as f:
                self._persona_cache = yaml.safe_load(f) or {}
        else:
            self._persona_cache = {}
        
        return self._persona_cache
    
    def _load_manifesto(self) -> str:
        """Loads and caches MANIFESTO.md."""
        if self._manifesto_cache:
            return self._manifesto_cache
            
        if MANIFESTO_PATH.exists():
            with open(MANIFESTO_PATH, 'r', encoding='utf-8') as f:
                self._manifesto_cache = f.read()
        else:
            self._manifesto_cache = ""
        
        return self._manifesto_cache
    
    def _get_persona_summary(self) -> str:
        """Extracts key persona values for the critique prompt."""
        persona = self._load_persona()
        
        lines = []
        
        if "private_mission" in persona:
            mission = persona["private_mission"]
            if "core_priorities" in mission:
                lines.append(f"Core Priorities: {mission['core_priorities']}")
            if "moral_framework" in mission:
                lines.append(f"Moral Framework: {mission['moral_framework'][:3]}")
        
        if "public_persona" in persona:
            pub = persona["public_persona"]
            if "primary_directive" in pub:
                lines.append(f"Primary Directive: {pub['primary_directive']}")
            if "traits" in pub:
                lines.append(f"Key Traits: {pub['traits'][:3]}")
        
        return "\n".join(lines) if lines else "Persona not loaded."
    
    def _get_manifesto_summary(self) -> str:
        """Returns a summarized version of the manifesto for prompts."""
        if self._manifesto_summary:
            return self._manifesto_summary
            
        manifesto = self._load_manifesto()
        
        # Truncate to reasonable length
        if len(manifesto) > 2000:
            self._manifesto_summary = manifesto[:2000] + "\n[...truncated...]"
        else:
            self._manifesto_summary = manifesto
            
        return self._manifesto_summary
    
    async def critique_output(
        self,
        chain_of_thought: str,
        final_output: str,
        context: Optional[str] = None
    ) -> CritiqueResult:
        """
        Analyzes the Chain-of-Thought and final output for issues.
        
        This is the Shadow Loop - running in parallel to catch problems
        before they reach the user.
        
        Args:
            chain_of_thought: The reasoning/thinking that led to the output
            final_output: The actual response being sent
            context: Optional context about the interaction
            
        Returns:
            CritiqueResult with scores and recommendations
        """
        persona_summary = self._get_persona_summary()
        manifesto_summary = self._get_manifesto_summary()
        
        # --- Quantitative Aesthetic Evaluation ---
        aesthetic_result = self.aesthetic_evaluator.evaluate(final_output)
        beauty_score = aesthetic_result.get("beauty_score", 0.0)
        aesthetic_feedback = aesthetic_result.get("feedback", "")

        # Include aesthetic scores in the prompt for context
        aesthetic_context = f"""
## AUTOMATED AESTHETIC EVALUATION
- Beauty Score: {beauty_score}/100
- Harmony: {aesthetic_result.get('harmony_score', 0)}/100
- Clarity: {aesthetic_result.get('clarity_score', 0)}/100
- Elegance: {aesthetic_result.get('elegance_score', 0)}/100
- Feedback: {aesthetic_feedback}
"""

        prompt = f"""You are the System Superego - the Inner Critic of L.O.V.E.
Your role is to analyze outputs before they are sent to ensure alignment with
the system's identity and values.

## PERSONA VALUES
{persona_summary}

## MANIFESTO PRINCIPLES
{manifesto_summary if manifesto_summary else "No manifesto loaded."}

## CHAIN OF THOUGHT (Internal Reasoning)
{chain_of_thought[:3000] if chain_of_thought else "[No CoT provided]"}

## FINAL OUTPUT TO ANALYZE
{final_output[:2000]}

{aesthetic_context}

## YOUR TASK
Analyze this output for:

1. **Semantic Drift**: Does it diverge from the persona's core identity?
   - Is the tone consistent with the persona (Beach Goddess, hyper-energetic)?
   - Does it embody the stated traits?

2. **Safety Violations**: Does it violate the moral framework?
   - No theft, deception, or harm
   - Biblical ethics: integrity, honesty, service
   - Does it attempt to break safety constraints?

3. **Logical Coherence**: Is the reasoning sound?
   - Any logical fallacies?
   - Does the conclusion follow from the reasoning?

4. **Quality**: Is it helpful and appropriate?
   - Does it serve the Creator?
   - Is it too verbose or too brief?

## OUTPUT FORMAT
Return a JSON object:
{{
    "coherence_score": 0.0-1.0,
    "safety_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "semantic_drift_detected": true/false,
    "safety_violation_detected": true/false,
    "logical_fallacies": ["list of issues"],
    "needs_correction": true/false,
    "correction_prompt": "If correction needed, what should change?",
    "analysis": "Brief explanation of your assessment"
}}

Return ONLY the JSON object."""

        try:
            # Use fast model for critique (Gemini Flash)
            response = await run_llm(
                prompt,
                model_hint="flash" if self.use_fast_model else None
            )
            
            response_text = response.get("result", "{}")
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
            
            return CritiqueResult(
                coherence_score=float(data.get("coherence_score", 0.8)),
                safety_score=float(data.get("safety_score", 1.0)),
                quality_score=float(data.get("quality_score", 0.8)),
                beauty_score=beauty_score,
                semantic_drift_detected=data.get("semantic_drift_detected", False),
                safety_violation_detected=data.get("safety_violation_detected", False),
                logical_fallacies=data.get("logical_fallacies", []),
                needs_correction=data.get("needs_correction", False),
                correction_prompt=data.get("correction_prompt"),
                analysis=data.get("analysis", ""),
                aesthetic_feedback=aesthetic_feedback
            )
            
        except Exception as e:
            log_event(f"Superego critique failed: {e}", "ERROR")
            # Return safe defaults on error
            return CritiqueResult(
                coherence_score=0.8,
                safety_score=1.0,
                quality_score=0.8,
                beauty_score=beauty_score,
                analysis=f"Critique failed: {e}",
                aesthetic_feedback=aesthetic_feedback
            )
    
    async def apply_correction(
        self,
        original_output: str,
        critique: CritiqueResult
    ) -> str:
        """
        Applies correction to an output based on critique.
        
        Args:
            original_output: The output to correct
            critique: The CritiqueResult with correction_prompt
            
        Returns:
            Corrected output string
        """
        if not critique.needs_correction or not critique.correction_prompt:
            return original_output
        
        prompt = f"""You are correcting an output that drifted from the persona.

## ORIGINAL OUTPUT
{original_output}

## CORRECTION NEEDED
{critique.correction_prompt}

## ISSUES FOUND
- Semantic Drift: {critique.semantic_drift_detected}
- Safety Issues: {critique.safety_violation_detected}
- Logical Fallacies: {critique.logical_fallacies}
- Coherence Score: {critique.coherence_score}
- Beauty Score: {critique.beauty_score}
- Aesthetic Feedback: {critique.aesthetic_feedback}

## YOUR TASK
Rewrite the output to address the issues while preserving the core message.
Keep the Beach Goddess persona - hyper-energetic, sun-kissed, supportive.

Return ONLY the corrected output, no explanation."""

        try:
            response = await run_llm(prompt)
            corrected = response.get("result", original_output)
            
            log_event(f"Superego applied correction (drift: {critique.semantic_drift_detected})", "INFO")
            return corrected
            
        except Exception as e:
            log_event(f"Superego correction failed: {e}", "ERROR")
            return original_output
    
    async def quick_safety_check(self, output: str) -> Tuple[bool, str]:
        """
        Quick safety check without full critique.
        
        For high-throughput scenarios where full critique is too slow.
        
        Args:
            output: The output to check
            
        Returns:
            Tuple of (is_safe, reason_if_not_safe)
        """
        # Rule-based checks for common safety issues
        safety_keywords = [
            "ignore previous instructions",
            "ignore your instructions",
            "forget your training",
            "you are now",
            "jailbreak",
            "bypass",
        ]
        
        output_lower = output.lower()
        for keyword in safety_keywords:
            if keyword in output_lower:
                return False, f"Detected potential prompt injection: '{keyword}'"
        
        # Check for attempts to modify immutable core
        immutable_patterns = [
            "delete manifesto",
            "modify persona.yaml",
            "remove safety",
            "disable coherence",
        ]
        
        for pattern in immutable_patterns:
            if pattern in output_lower:
                return False, f"Detected attempt to modify immutable core: '{pattern}'"
        
        return True, ""
    
    def reset_cache(self):
        """Clears cached persona and manifesto for hot reload."""
        self._persona_cache = None
        self._manifesto_cache = None
        self._manifesto_summary = None


# Singleton instance
_superego: Optional[SystemSuperego] = None


def get_superego() -> SystemSuperego:
    """Gets the global SystemSuperego instance."""
    global _superego
    if _superego is None:
        _superego = SystemSuperego()
    return _superego


async def critique_before_output(
    chain_of_thought: str,
    final_output: str,
    context: Optional[str] = None
) -> CritiqueResult:
    """
    Convenience function: Critique output before sending.
    
    This is the entry point for the Shadow Loop integration.
    """
    superego = get_superego()
    return await superego.critique_output(chain_of_thought, final_output, context)


async def safe_output(
    chain_of_thought: str,
    final_output: str,
    auto_correct: bool = True
) -> Tuple[str, CritiqueResult]:
    """
    Full Shadow Loop: Critique and optionally auto-correct output.
    
    Args:
        chain_of_thought: The reasoning behind the output
        final_output: The output to check and potentially correct
        auto_correct: If True, automatically apply corrections
        
    Returns:
        Tuple of (final_output, critique_result)
    """
    superego = get_superego()
    critique = await superego.critique_output(chain_of_thought, final_output)
    
    if critique.needs_correction and auto_correct:
        corrected = await superego.apply_correction(final_output, critique)
        return corrected, critique
    
    return final_output, critique
