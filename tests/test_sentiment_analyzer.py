"""
Tests for Story 3.1: Context-Aware Tone Shift
"""
import pytest


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""
    
    def test_analyze_positive_text(self):
        """Test analyzing positive text."""
        from core.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I love this! Amazing work, you're incredible! 💖✨")
        
        assert result.dominant == "positive"
        assert result.positive_score > result.negative_score
    
    def test_analyze_negative_text(self):
        """Test analyzing negative text."""
        from core.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This is stupid fake garbage, you're pathetic")
        
        assert result.dominant == "negative"
        assert result.negative_score > result.positive_score
    
    def test_analyze_neutral_text(self):
        """Test analyzing neutral text."""
        from core.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I saw your post today.")
        
        assert result.dominant == "neutral"
    
    def test_emoji_boost_positive(self):
        """Test that positive emojis boost positive score."""
        from core.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("❤💖😍🔥✨")
        
        assert result.positive_score > 0


class TestToneParameters:
    """Tests for ToneParameters class."""
    
    def test_tone_to_prompt_text(self):
        """Test conversion to prompt guidance."""
        from core.sentiment_analyzer import ToneParameters, ToneStyle
        
        tone = ToneParameters(
            warmth=0.9,
            assertiveness=0.4,
            playfulness=0.6,
            style=ToneStyle.WARM_APPRECIATIVE
        )
        
        text = tone.to_prompt_text()
        
        assert "TONE GUIDANCE" in text
        assert "warm" in text.lower()
        assert "0.9" in text


class TestGetToneForInteraction:
    """Tests for tone selection based on user type."""
    
    def test_creator_gets_reverent_tone(self):
        """Test that Creator always gets reverent tone."""
        from core.sentiment_analyzer import SentimentAnalyzer, SentimentResult, ToneStyle
        
        analyzer = SentimentAnalyzer()
        sentiment = SentimentResult(0.5, 0.5, 0.0, "neutral", 0.5)
        
        tone = analyzer.get_tone_for_interaction("Creator", sentiment)
        
        assert tone.style == ToneStyle.REVERENT_DEVOTED
        assert tone.warmth == 1.0
    
    def test_hater_high_intensity_gets_witty(self):
        """Test that strong haters get witty dismissive tone."""
        from core.sentiment_analyzer import SentimentAnalyzer, SentimentResult, ToneStyle
        
        analyzer = SentimentAnalyzer()
        sentiment = SentimentResult(0.0, 0.8, 0.2, "negative", 0.8)
        
        tone = analyzer.get_tone_for_interaction("Hater", sentiment)
        
        assert tone.style == ToneStyle.WITTY_DISMISSIVE
        assert tone.assertiveness >= 0.7
    
    def test_hater_low_intensity_gets_stoic(self):
        """Test that mild haters get high road stoic tone."""
        from core.sentiment_analyzer import SentimentAnalyzer, SentimentResult, ToneStyle
        
        analyzer = SentimentAnalyzer()
        sentiment = SentimentResult(0.1, 0.3, 0.6, "neutral", 0.3)
        
        tone = analyzer.get_tone_for_interaction("Hater", sentiment)
        
        assert tone.style == ToneStyle.HIGH_ROAD_STOIC
    
    def test_fan_positive_gets_warm(self):
        """Test that enthusiastic fans get warm appreciative tone."""
        from core.sentiment_analyzer import SentimentAnalyzer, SentimentResult, ToneStyle
        
        analyzer = SentimentAnalyzer()
        sentiment = SentimentResult(0.9, 0.0, 0.1, "positive", 0.8)
        
        tone = analyzer.get_tone_for_interaction("Fan", sentiment)
        
        assert tone.style == ToneStyle.WARM_APPRECIATIVE
        assert tone.warmth >= 0.8


class TestConvenienceFunction:
    """Tests for analyze_and_get_tone convenience function."""
    
    def test_analyze_and_get_tone_returns_tuple(self):
        """Test that function returns both sentiment and tone."""
        from core.sentiment_analyzer import analyze_and_get_tone
        
        sentiment, tone = analyze_and_get_tone("I love your work! ✨", "Fan")
        
        assert sentiment is not None
        assert tone is not None
        assert hasattr(sentiment, "dominant")
        assert hasattr(tone, "style")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
