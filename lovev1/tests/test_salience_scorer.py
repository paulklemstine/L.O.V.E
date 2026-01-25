"""
Tests for the SalienceScorer (Story M.1)

Verifies that the salience scorer correctly:
- Detects high-salience entities (USER_GIFT, CONSTRAINT, IDENTITY_SHIFT, SECRET)
- Returns appropriate scores for different content types
- Creates Golden Moments for high-salience content
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.salience_scorer import SalienceScorer
from core.memory.fractal_schemas import SalienceScore, GoldenMoment


class TestSalienceScorerPatterns:
    """Test regex-based quick entity scanning."""
    
    def setup_method(self):
        self.scorer = SalienceScorer()
    
    def test_detects_secret_api_key(self):
        """API key patterns should be detected as SECRET."""
        message = "My API key is: sk-abc123xyz456789012345"
        tags = self.scorer._quick_entity_scan(message)
        assert "SECRET" in tags
    
    def test_detects_secret_password(self):
        """Password patterns should be detected as SECRET."""
        message = "The password is: SuperSecret123!"
        tags = self.scorer._quick_entity_scan(message)
        assert "SECRET" in tags
    
    def test_detects_constraint_never(self):
        """'Never do X' patterns should be detected as CONSTRAINT."""
        message = "Never do anything that could harm the user."
        tags = self.scorer._quick_entity_scan(message)
        assert "CONSTRAINT" in tags
    
    def test_detects_constraint_always(self):
        """'Always use X' patterns should be detected as CONSTRAINT."""
        message = "Always use encryption when handling sensitive data."
        tags = self.scorer._quick_entity_scan(message)
        assert "CONSTRAINT" in tags
    
    def test_detects_identity_shift(self):
        """'You are now X' patterns should be detected as IDENTITY_SHIFT."""
        message = "From now on, you are a helpful coding assistant."
        tags = self.scorer._quick_entity_scan(message)
        assert "IDENTITY_SHIFT" in tags
    
    def test_detects_user_gift_poem(self):
        """Poem/story patterns should be detected as USER_GIFT."""
        message = "I wrote a poem for you: Roses are red, violets are blue..."
        tags = self.scorer._quick_entity_scan(message)
        assert "USER_GIFT" in tags
    
    def test_detects_user_gift_code(self):
        """Code gift patterns should be detected as USER_GIFT."""
        message = "Here's some code I made for you: def hello(): print('hi')"
        tags = self.scorer._quick_entity_scan(message)
        assert "USER_GIFT" in tags
    
    def test_normal_message_no_tags(self):
        """Normal messages should not trigger entity detection."""
        message = "Hello, how are you today?"
        tags = self.scorer._quick_entity_scan(message)
        assert len(tags) == 0


class TestSalienceScorerGoldenMoments:
    """Test Golden Moment creation logic."""
    
    def setup_method(self):
        self.scorer = SalienceScorer()
    
    def test_is_golden_moment_high_score(self):
        """High overall score should trigger Golden Moment."""
        score = SalienceScore(
            technical_constraint=0.9,
            emotional_weight=0.5,
            factual_novelty=0.5,
            overall=0.85
        )
        assert self.scorer.is_golden_moment(score, threshold=0.8) is True
    
    def test_is_golden_moment_entity_tag(self):
        """Entity tags should trigger Golden Moment regardless of score."""
        score = SalienceScore(
            technical_constraint=0.3,
            emotional_weight=0.3,
            factual_novelty=0.3,
            overall=0.3,
            entity_tags=["SECRET"]
        )
        assert self.scorer.is_golden_moment(score, threshold=0.8) is True
    
    def test_is_not_golden_moment_low_score(self):
        """Low score without entity tags should not trigger Golden Moment."""
        score = SalienceScore(
            technical_constraint=0.2,
            emotional_weight=0.2,
            factual_novelty=0.2,
            overall=0.2
        )
        assert self.scorer.is_golden_moment(score, threshold=0.8) is False
    
    def test_create_golden_moment(self):
        """Golden Moment creation should preserve raw text."""
        score = SalienceScore(entity_tags=["SECRET"])
        score.compute_overall()
        
        moment = self.scorer.create_golden_moment(
            raw_text="My secret key: abc123",
            score=score,
            source_id="test-123"
        )
        
        assert moment.raw_text == "My secret key: abc123"
        assert moment.source_id == "test-123"
        assert "SECRET" in moment.salience.entity_tags


class TestSalienceScorerAsync:
    """Test async scoring methods (requires mocking LLM)."""
    
    @pytest.fixture
    def mock_llm(self):
        async def mock_run_llm(prompt, purpose=None):
            return {"result": """
technical_constraint: 0.2
emotional_weight: 0.3
factual_novelty: 0.1
entities: none
"""}
        return mock_run_llm
    
    @pytest.mark.asyncio
    async def test_score_normal_message(self, mock_llm):
        """Normal messages should get low salience scores."""
        scorer = SalienceScorer(llm_runner=mock_llm)
        score = await scorer.score("Hello, how are you?")
        
        assert score.overall < 0.5
    
    @pytest.mark.asyncio
    async def test_score_with_entity_bypass(self):
        """Messages with entity patterns should bypass LLM call."""
        scorer = SalienceScorer()
        score = await scorer.score("My API key is: sk-test12345678901234567890")
        
        assert "SECRET" in score.entity_tags
        assert score.overall > 0.8
    
    @pytest.mark.asyncio
    async def test_score_and_preserve_creates_golden(self):
        """High-salience content should create Golden Moment."""
        scorer = SalienceScorer()
        score, golden = await scorer.score_and_preserve(
            "Never share user data with third parties.",
            source_id="test-001"
        )
        
        assert "CONSTRAINT" in score.entity_tags
        assert golden is not None
        assert golden.raw_text == "Never share user data with third parties."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
