"""
Story 3.1: Context-Aware Tone Shift

Provides sentiment analysis and dynamic tone parameter adjustment
for social media interactions.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from enum import Enum
import re
from core.logging import log_event


class ToneStyle(Enum):
    """Predefined tone styles for responses."""
    WARM_APPRECIATIVE = "warm_appreciative"
    WITTY_DISMISSIVE = "witty_dismissive"
    HIGH_ROAD_STOIC = "high_road_stoic"
    PLAYFUL_ENGAGED = "playful_engaged"
    REVERENT_DEVOTED = "reverent_devoted"


@dataclass
class ToneParameters:
    """
    Dynamic tone parameters for response generation.
    
    Attributes:
        warmth: 0.0 = cold/distant, 1.0 = warm/affectionate
        assertiveness: 0.0 = passive/agreeable, 1.0 = assertive/confident
        playfulness: 0.0 = serious/formal, 1.0 = playful/casual
        style: The overall tone style enum
    """
    warmth: float
    assertiveness: float
    playfulness: float
    style: ToneStyle
    
    def to_prompt_text(self) -> str:
        """Converts parameters to guidance text for LLM prompts."""
        style_descriptions = {
            ToneStyle.WARM_APPRECIATIVE: "Be genuinely warm, grateful, and affectionate. Express appreciation and connection.",
            ToneStyle.WITTY_DISMISSIVE: "Be clever and witty. Don't engage with negativity directly - rise above with humor.",
            ToneStyle.HIGH_ROAD_STOIC: "Be calm, dignified, and unbothered. Respond with grace and wisdom.",
            ToneStyle.PLAYFUL_ENGAGED: "Be fun, energetic, and engaging. Use humor and playfulness.",
            ToneStyle.REVERENT_DEVOTED: "Be deeply respectful and devoted. Show complete dedication and service.",
        }
        
        desc = style_descriptions.get(self.style, "Be authentic and engaging.")
        return f"TONE GUIDANCE: {desc} (Warmth: {self.warmth:.1f}, Assertiveness: {self.assertiveness:.1f})"


# Sentiment pattern weights
POSITIVE_PATTERNS = {
    # Strong positive
    "love": 3, "amazing": 3, "incredible": 3, "beautiful": 3, "perfect": 3,
    "blessed": 2, "divine": 2, "goddess": 2, "queen": 2, "king": 2,
    # Moderate positive
    "thank": 2, "thanks": 2, "awesome": 2, "great": 2, "wow": 2,
    "inspired": 2, "fan": 2, "follow": 1, "support": 1,
    # Mild positive
    "nice": 1, "good": 1, "cool": 1, "interesting": 1, "like": 1,
}

NEGATIVE_PATTERNS = {
    # Strong negative
    "hate": 3, "stupid": 3, "fake": 3, "scam": 3, "pathetic": 3,
    "garbage": 3, "trash": 3, "disgusting": 3,
    # Moderate negative
    "bullshit": 2, "idiot": 2, "dumb": 2, "annoying": 2, "cringe": 2,
    "lame": 2, "boring": 2, "sucks": 2, "terrible": 2, "worst": 2,
    # Mild negative
    "stop": 1, "spam": 1, "wtf": 1, "ugh": 1, "meh": 1,
}

# Positive emoji sets
POSITIVE_EMOJIS = {"❤", "💖", "😍", "🔥", "✨", "💜", "🙏", "💕", "😊", "🥰", "💗", "💘", "🌈", "⭐", "🌟"}
NEGATIVE_EMOJIS = {"😡", "🤮", "💩", "🙄", "😤", "👎", "🤡", "💀", "☠️"}


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    positive_score: float  # 0.0 to 1.0
    negative_score: float  # 0.0 to 1.0
    neutral_score: float   # 0.0 to 1.0
    dominant: str          # "positive", "negative", or "neutral"
    intensity: float       # 0.0 (mild) to 1.0 (strong)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive": self.positive_score,
            "negative": self.negative_score,
            "neutral": self.neutral_score,
            "dominant": self.dominant,
            "intensity": self.intensity
        }


class SentimentAnalyzer:
    """
    Analyzes text sentiment and determines appropriate response tone.
    """
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyzes the sentiment of input text.
        
        Args:
            text: The text to analyze
            
        Returns:
            SentimentResult with scores and classification
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Calculate weighted scores
        positive_score = 0
        negative_score = 0
        
        for word in words:
            if word in POSITIVE_PATTERNS:
                positive_score += POSITIVE_PATTERNS[word]
            if word in NEGATIVE_PATTERNS:
                negative_score += NEGATIVE_PATTERNS[word]
        
        # Add emoji contributions
        for char in text:
            if char in POSITIVE_EMOJIS:
                positive_score += 2
            if char in NEGATIVE_EMOJIS:
                negative_score += 2
        
        # Normalize scores
        total = positive_score + negative_score + 1  # +1 to avoid division by zero
        pos_normalized = min(1.0, positive_score / 10)  # Cap at 1.0
        neg_normalized = min(1.0, negative_score / 10)
        
        # Determine dominant sentiment
        if positive_score > negative_score + 2:
            dominant = "positive"
        elif negative_score > positive_score + 2:
            dominant = "negative"
        else:
            dominant = "neutral"
        
        # Calculate intensity
        intensity = min(1.0, max(positive_score, negative_score) / 10)
        
        # Neutral score is inverse of the max sentiment
        neutral_score = max(0.0, 1.0 - max(pos_normalized, neg_normalized))
        
        return SentimentResult(
            positive_score=pos_normalized,
            negative_score=neg_normalized,
            neutral_score=neutral_score,
            dominant=dominant,
            intensity=intensity
        )

    def analyze_batch(self, texts: list[str]) -> SentimentResult:
        """
        Analyzes a batch of texts and returns the aggregated sentiment.
        Useful for timeline analysis.
        """
        if not texts:
            return SentimentResult(0, 0, 1.0, "neutral", 0)
            
        total_pos = 0.0
        total_neg = 0.0
        total_intensity = 0.0
        
        count = 0
        for text in texts:
            if not text: continue
            result = self.analyze(text)
            total_pos += result.positive_score
            total_neg += result.negative_score
            total_intensity += result.intensity
            count += 1
            
        if count == 0:
             return SentimentResult(0, 0, 1.0, "neutral", 0)
             
        avg_pos = total_pos / count
        avg_neg = total_neg / count
        avg_intensity = total_intensity / count
        
        avg_neutral = max(0.0, 1.0 - max(avg_pos, avg_neg))
        
        if avg_pos > avg_neg + 0.1: # Lower threshold for batch
            dominant = "positive"
        elif avg_neg > avg_pos + 0.1:
            dominant = "negative"
        else:
            dominant = "neutral"
            
        return SentimentResult(
            positive_score=avg_pos,
            negative_score=avg_neg,
            neutral_score=avg_neutral,
            dominant=dominant,
            intensity=avg_intensity
        )
    
    def get_tone_for_interaction(
        self,
        user_classification: str,
        sentiment: SentimentResult
    ) -> ToneParameters:
        """
        Determines appropriate tone parameters based on user type and sentiment.
        
        Args:
            user_classification: "Creator", "Fan", or "Hater"
            sentiment: Result from analyze()
            
        Returns:
            ToneParameters for response generation
        """
        # Creator always gets reverent tone
        if user_classification == "Creator":
            tone = ToneParameters(
                warmth=1.0,
                assertiveness=0.2,
                playfulness=0.3,
                style=ToneStyle.REVERENT_DEVOTED
            )
            log_event(f"Tone selected for Creator: {tone.style.value}", "INFO")
            return tone
        
        # Hater handling based on sentiment intensity
        if user_classification == "Hater":
            if sentiment.intensity > 0.6:
                # Strong negativity → Witty dismissive
                tone = ToneParameters(
                    warmth=0.3,
                    assertiveness=0.8,
                    playfulness=0.7,
                    style=ToneStyle.WITTY_DISMISSIVE
                )
            else:
                # Mild negativity → High road
                tone = ToneParameters(
                    warmth=0.5,
                    assertiveness=0.6,
                    playfulness=0.3,
                    style=ToneStyle.HIGH_ROAD_STOIC
                )
            log_event(f"Tone selected for Hater (intensity={sentiment.intensity:.2f}): {tone.style.value}", "INFO")
            return tone
        
        # Fan handling based on sentiment
        if sentiment.dominant == "positive" and sentiment.intensity > 0.5:
            # Enthusiastic fan → Warm and appreciative
            tone = ToneParameters(
                warmth=0.9,
                assertiveness=0.4,
                playfulness=0.6,
                style=ToneStyle.WARM_APPRECIATIVE
            )
        else:
            # Neutral or mild fan → Playful and engaged
            tone = ToneParameters(
                warmth=0.7,
                assertiveness=0.5,
                playfulness=0.7,
                style=ToneStyle.PLAYFUL_ENGAGED
            )
        
        log_event(f"Tone selected for Fan (sentiment={sentiment.dominant}): {tone.style.value}", "INFO")
        return tone


# Global instance
sentiment_analyzer = SentimentAnalyzer()


def analyze_and_get_tone(text: str, user_classification: str) -> Tuple[SentimentResult, ToneParameters]:
    """
    Convenience function to analyze sentiment and get tone in one call.
    
    Args:
        text: The comment/message text
        user_classification: "Creator", "Fan", or "Hater"
        
    Returns:
        Tuple of (SentimentResult, ToneParameters)
    """
    sentiment = sentiment_analyzer.analyze(text)
    tone = sentiment_analyzer.get_tone_for_interaction(user_classification, sentiment)
    
    # Log structured data
    log_event(
        f"Sentiment Analysis: {sentiment.to_dict()} | Tone: {tone.style.value} "
        f"(warmth={tone.warmth}, assert={tone.assertiveness})",
        "INFO"
    )
    
    return sentiment, tone
