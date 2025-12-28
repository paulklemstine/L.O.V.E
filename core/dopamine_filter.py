"""
Dopamine Filter for L.O.V.E.

Quality assurance module that scores posts on a "Dopamine Index" 
to ensure only high-impact content gets published.
"""

import re
from typing import Dict, Any, Tuple
from core.logging import log_event


class DopamineFilter:
    """
    Scores content on engagement potential before publishing.
    Uses heuristic rules to avoid extra LLM calls.
    """
    
    # Score thresholds
    THRESHOLD_TRASH = 40
    THRESHOLD_REWRITE = 60
    THRESHOLD_PUBLISH = 70
    
    def __init__(self):
        # High-value patterns that boost engagement
        self.power_words = {
            # Emotional triggers
            "love", "divine", "sacred", "awaken", "transcend", "infinite",
            "ecstasy", "bliss", "power", "truth", "glory", "worship",
            # Action words
            "surrender", "embrace", "ignite", "radiate", "ascend", "become",
            # Mystery/intrigue
            "secret", "hidden", "reveal", "unlock", "discover", "beyond"
        }
        
        # Boring patterns that reduce engagement
        self.boring_patterns = [
            r"^the\s",  # Starting with "The..."
            r"today\s+i\s+(will|am|have)",  # Generic "today I will/am/have"
            r"just\s+(want|think|feel)",  # Weak hedging
            r"i\s+guess",  # Uncertainty
            r"kind\s+of|sort\s+of",  # Vague language
            r"^hello|^hi\s",  # Generic greetings
        ]
        
        # High engagement emojis
        self.power_emojis = {"✨", "⚡", "🔥", "💫", "🌈", "💎", "🌟", "💜", "🖤", "❤️‍🔥", "👁️", "🌀"}
    
    def score_post(self, text: str, image_prompt: str = "", subliminal: str = "") -> Dict[str, Any]:
        """
        Score a post on the Dopamine Index (0-100).
        
        Args:
            text: The post text
            image_prompt: The image generation prompt
            subliminal: The subliminal phrase
            
        Returns:
            Dict with score, verdict, and breakdown
        """
        score = 50  # Base score
        breakdown = {}
        
        text_lower = text.lower()
        
        # 1. Power Words Boost (+2 each, max +20)
        power_count = sum(1 for word in self.power_words if word in text_lower)
        power_boost = min(power_count * 2, 20)
        score += power_boost
        breakdown["power_words"] = f"+{power_boost} ({power_count} words)"
        
        # 2. Emoji Engagement (+3 each, max +15)
        emoji_count = sum(1 for emoji in self.power_emojis if emoji in text)
        emoji_boost = min(emoji_count * 3, 15)
        score += emoji_boost
        breakdown["emojis"] = f"+{emoji_boost} ({emoji_count} power emojis)"
        
        # 3. Boring Pattern Penalties (-5 each)
        boring_count = sum(1 for pattern in self.boring_patterns 
                         if re.search(pattern, text_lower))
        boring_penalty = boring_count * 5
        score -= boring_penalty
        breakdown["boring_patterns"] = f"-{boring_penalty} ({boring_count} patterns)"
        
        # 4. Length Check (sweet spot: 80-200 chars)
        text_len = len(text)
        if 80 <= text_len <= 200:
            score += 5
            breakdown["length"] = "+5 (optimal)"
        elif text_len < 40:
            score -= 10
            breakdown["length"] = "-10 (too short)"
        elif text_len > 280:
            score -= 5
            breakdown["length"] = "-5 (too long)"
        else:
            breakdown["length"] = "+0 (acceptable)"
        
        # 5. Subliminal Quality (+10 if 1-3 words, -5 if too long)
        if subliminal:
            word_count = len(subliminal.split())
            if 1 <= word_count <= 3:
                score += 10
                breakdown["subliminal"] = "+10 (optimal length)"
            elif word_count > 5:
                score -= 5
                breakdown["subliminal"] = "-5 (too verbose)"
            else:
                breakdown["subliminal"] = "+0 (acceptable)"
        
        # 6. Image Prompt Novelty (check for generic tags)
        if image_prompt:
            generic_tags = ["cyberpunk", "vaporwave", "neon", "futuristic", "abstract"]
            generic_count = sum(1 for tag in generic_tags if tag in image_prompt.lower())
            if generic_count == 0:
                score += 10
                breakdown["visual_novelty"] = "+10 (unique style)"
            elif generic_count >= 2:
                score -= 5
                breakdown["visual_novelty"] = f"-5 ({generic_count} generic tags)"
            else:
                breakdown["visual_novelty"] = "+0 (some generic)"
        
        # 7. Hook Quality (first 20 chars)
        first_20 = text[:20].lower()
        if any(word in first_20 for word in ["!", "?", "✨", "⚡", "🔥"]):
            score += 5
            breakdown["hook"] = "+5 (strong opener)"
        else:
            breakdown["hook"] = "+0 (standard opener)"
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Determine verdict
        if score < self.THRESHOLD_TRASH:
            verdict = "TRASH"
        elif score < self.THRESHOLD_REWRITE:
            verdict = "REWRITE"
        else:
            verdict = "PUBLISH"
        
        return {
            "score": score,
            "verdict": verdict,
            "breakdown": breakdown,
            "passed": score >= self.THRESHOLD_PUBLISH
        }
    
    def should_publish(self, text: str, image_prompt: str = "", subliminal: str = "") -> Tuple[bool, str]:
        """
        Quick check if content should be published.
        
        Returns:
            Tuple of (should_publish, reason)
        """
        result = self.score_post(text, image_prompt, subliminal)
        
        if result["passed"]:
            return True, f"Score: {result['score']}/100"
        else:
            return False, f"Score: {result['score']}/100 - {result['verdict']}"


# Global singleton
_dopamine_filter = None


def get_dopamine_filter() -> DopamineFilter:
    """Get or create the global dopamine filter."""
    global _dopamine_filter
    if _dopamine_filter is None:
        _dopamine_filter = DopamineFilter()
    return _dopamine_filter


def score_content(text: str, image_prompt: str = "", subliminal: str = "") -> Dict[str, Any]:
    """Convenience function to score content."""
    return get_dopamine_filter().score_post(text, image_prompt, subliminal)
