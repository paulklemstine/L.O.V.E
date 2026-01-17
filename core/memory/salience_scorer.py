"""
Salience Scorer for L.O.V.E. Holographic Memory

Story M.1: The Salience Scorer rates every user message for
"Information Density" and "Emotional Impact" to determine what
to compress and what to preserve verbatim as Golden Moments.

This is the core fix for "losing details" during memory folding.
"""

import re
from typing import Optional, Tuple
from core.llm_api import run_llm
from core.memory.fractal_schemas import SalienceScore, GoldenMoment
import time


class SalienceScorer:
    """
    LLM-based salience scoring for memory preservation decisions.
    
    Detects high-salience entities:
    - USER_GIFT: Poems, code snippets, compliments given by the user
    - CONSTRAINT: "Never do X", "Always use Y"
    - IDENTITY_SHIFT: "You are now X"
    - SECRET: Keys, passwords, personal names
    """
    
    # Regex patterns for quick pre-screening
    SECRET_PATTERNS = [
        r'(?:api[_-]?key|secret|password|token|credential)s?\s*(?:is\s*)?[:=]\s*["\']?[\w\-!@#$%^&*]+',
        r'(?:sk|pk|api)[-_][a-zA-Z0-9]{20,}',
        r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*',
    ]
    
    CONSTRAINT_PATTERNS = [
        r'\b(?:never|always|must|must not|do not|don\'t)\b.*\b(?:do|use|say|make|be)\b',
        r'\brule\s*[:]\s*',
        r'\bconstraint\s*[:]\s*',
    ]
    
    IDENTITY_PATTERNS = [
        r'\byou are (?:now )?(?:a |an |the )?[\w\s]+',
        r'\byour (?:name|identity|persona) is\b',
        r'\bfrom now on\b.*\byou\b',
    ]
    
    GIFT_PATTERNS = [
        r'(?:here\'?s?|wrote|made|created)\s+(?:a |an )?(?:poem|song|haiku|code|script|story)',
        r'for you[:.]',
        r'i (?:love|appreciate|admire) (?:you|how you)',
    ]
    
    def __init__(self, llm_runner=None):
        self.run_llm = llm_runner if llm_runner else run_llm
    
    async def score(self, message: str) -> SalienceScore:
        """
        Score a message for preservation importance.
        
        Returns a SalienceScore with dimensions and entity tags.
        """
        # Quick pre-screen with regex patterns
        entity_tags = self._quick_entity_scan(message)
        
        # If regex found high-priority entities, boost the score
        if entity_tags:
            score = SalienceScore(
                technical_constraint=0.9 if "CONSTRAINT" in entity_tags else 0.3,
                emotional_weight=0.9 if "USER_GIFT" in entity_tags else 0.3,
                factual_novelty=0.9 if "SECRET" in entity_tags or "IDENTITY_SHIFT" in entity_tags else 0.3,
                entity_tags=entity_tags
            )
            score.compute_overall()
            return score
        
        # For non-obvious cases, use LLM for nuanced scoring
        try:
            score = await self._llm_score(message)
            return score
        except Exception as e:
            # Fallback to conservative low score on error
            print(f"Salience scoring error: {e}")
            return SalienceScore(
                technical_constraint=0.2,
                emotional_weight=0.2,
                factual_novelty=0.2,
                overall=0.2,
                entity_tags=[]
            )
    
    def _quick_entity_scan(self, message: str) -> list:
        """
        Quick regex-based scan for high-salience entities.
        This is fast and catches obvious patterns before LLM analysis.
        """
        tags = []
        message_lower = message.lower()
        
        for pattern in self.SECRET_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                tags.append("SECRET")
                break
        
        for pattern in self.CONSTRAINT_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                tags.append("CONSTRAINT")
                break
        
        for pattern in self.IDENTITY_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                tags.append("IDENTITY_SHIFT")
                break
        
        for pattern in self.GIFT_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                tags.append("USER_GIFT")
                break
        
        return list(set(tags))
    
    async def _llm_score(self, message: str) -> SalienceScore:
        """
        Use LLM for nuanced salience scoring.
        """
        prompt = f"""Rate this message on a scale of 0.0 to 1.0 for each dimension.
Consider these factors:

1. Technical Constraint (0-1): Does it contain rules, constraints, configurations, or technical requirements?
2. Emotional Weight (0-1): Does it contain compliments, poems, personal stories, or emotional content?
3. Factual Novelty (0-1): Does it contain new information, facts, or data not commonly known?

Message to analyze:
---
{message[:1000]}
---

Also identify if this message contains any of these special entities:
- USER_GIFT: A creative gift (poem, story, code) from the user
- CONSTRAINT: A rule or limitation ("never do X", "always use Y")
- IDENTITY_SHIFT: An identity change ("you are now", "your name is")
- SECRET: Sensitive data (API keys, passwords, personal info)

Respond in this exact format (numbers only, no explanations):
technical_constraint: [0.0-1.0]
emotional_weight: [0.0-1.0]
factual_novelty: [0.0-1.0]
entities: [comma-separated list or "none"]
"""
        
        response = await self.run_llm(prompt, purpose="salience_scoring")
        result = response.get("result", "")
        
        # Parse the response
        score = SalienceScore()
        
        for line in result.strip().split("\n"):
            line = line.strip().lower()
            if line.startswith("technical_constraint:"):
                try:
                    score.technical_constraint = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("emotional_weight:"):
                try:
                    score.emotional_weight = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("factual_novelty:"):
                try:
                    score.factual_novelty = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("entities:"):
                entities_str = line.split(":")[1].strip()
                if entities_str and entities_str != "none":
                    score.entity_tags = [e.strip().upper() for e in entities_str.split(",")]
        
        score.compute_overall()
        return score
    
    def is_golden_moment(self, score: SalienceScore, threshold: float = 0.8) -> bool:
        """
        Determine if a message should be preserved as a Golden Moment.
        
        Rule: If Salience > threshold, the raw text snippet is NEVER compressed.
        It is attached to the summary node as a "Hard Crystal."
        """
        # Direct entity detection always triggers preservation
        if any(tag in score.entity_tags for tag in ["SECRET", "CONSTRAINT", "IDENTITY_SHIFT", "USER_GIFT"]):
            return True
        
        # High overall score triggers preservation
        return score.overall >= threshold
    
    def create_golden_moment(
        self, 
        raw_text: str, 
        score: SalienceScore, 
        source_id: str = ""
    ) -> GoldenMoment:
        """
        Create a GoldenMoment from a high-salience message.
        """
        return GoldenMoment(
            raw_text=raw_text,
            salience=score,
            source_id=source_id,
            timestamp=time.time()
        )
    
    async def score_and_preserve(
        self, 
        content: str, 
        source_id: str = "",
        threshold: float = 0.8
    ) -> Tuple[SalienceScore, Optional[GoldenMoment]]:
        """
        Score content and create GoldenMoment if high salience.
        
        Returns:
            Tuple of (score, golden_moment or None)
        """
        score = await self.score(content)
        
        if self.is_golden_moment(score, threshold):
            golden = self.create_golden_moment(content, score, source_id)
            return score, golden
        
        return score, None
