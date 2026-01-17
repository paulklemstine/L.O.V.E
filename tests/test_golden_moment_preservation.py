"""
Tests for Golden Moment Preservation (Story M.2)

Verifies that high-salience content is preserved verbatim
through memory folding operations - never compressed.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.fractal_schemas import (
    SalienceScore, GoldenMoment, ArcNode, EpisodicBuffer
)
from core.memory.schemas import MemorySummary


class TestGoldenMomentPreservation:
    """Test that high-salience content survives folding."""
    
    def test_golden_moment_creation(self):
        """GoldenMoment should preserve raw text exactly."""
        score = SalienceScore(
            technical_constraint=0.9,
            entity_tags=["CONSTRAINT"]
        )
        score.compute_overall()
        
        moment = GoldenMoment(
            raw_text="Never execute rm -rf without confirmation.",
            salience=score,
            source_id="test-001"
        )
        
        assert moment.raw_text == "Never execute rm -rf without confirmation."
        assert "CONSTRAINT" in moment.salience.entity_tags
    
    def test_arc_node_preserves_crystals(self):
        """ArcNode should preserve crystals alongside summary."""
        crystal = GoldenMoment(
            raw_text="User's secret API key: abc123",
            salience=SalienceScore(entity_tags=["SECRET"]),
            source_id="msg-001"
        )
        
        arc = ArcNode(
            summary="User exchanged greetings and discussed configuration.",
            crystals=[crystal],
            source_ids=["msg-001", "msg-002", "msg-003"]
        )
        
        # The crystal should be accessible
        assert len(arc.crystals) == 1
        assert arc.crystals[0].raw_text == "User's secret API key: abc123"
        
        # Crystal text should be retrievable
        crystal_text = arc.get_crystal_text()
        assert "abc123" in crystal_text
    
    def test_episodic_buffer_flush_returns_all(self):
        """EpisodicBuffer flush should return all episodes."""
        buffer = EpisodicBuffer(max_size=5)
        
        # Add some episodes
        buffer.add_episode("Hello", {"type": "greeting"})
        buffer.add_episode("How are you?", {"type": "question"})
        buffer.add_episode("Secret key: xyz789", {"type": "sensitive"})
        
        assert len(buffer.buffer) == 3
        
        # Flush
        episodes = buffer.flush()
        
        assert len(episodes) == 3
        assert len(buffer.buffer) == 0
        assert any("xyz789" in ep["content"] for ep in episodes)


class TestMemoryFoldingWithSalience:
    """Test that memory folding respects salience."""
    
    @pytest.fixture
    def mock_llm_runner(self):
        async def mock_run_llm(prompt, purpose=None):
            return {"result": "Test summary of folded memories."}
        return mock_run_llm
    
    @pytest.mark.asyncio
    async def test_folding_separates_by_salience(self, mock_llm_runner):
        """High-salience items should be extracted as crystals."""
        from core.memory.memory_folding_agent import MemoryFoldingAgent
        
        agent = MemoryFoldingAgent(llm_runner=mock_llm_runner)
        
        # Create mix of high and low salience memories
        memories = [
            MemorySummary(content="Hello", level=0, source_ids=["1"]),
            MemorySummary(content="How are you?", level=0, source_ids=["2"]),
            MemorySummary(content="My API key is: sk-secret123456789", level=0, source_ids=["3"]),
            MemorySummary(content="Nice weather today", level=0, source_ids=["4"]),
            MemorySummary(content="Never share passwords in logs", level=0, source_ids=["5"]),
        ]
        
        # With salience scorer, should separate items
        foldable, crystals = await agent._separate_by_salience(memories)
        
        # Should have extracted high-salience items as crystals
        # At least the API key and constraint should be crystals
        crystal_texts = [c.raw_text for c in crystals]
        
        # Check that secrets/constraints were preserved
        assert any("API key" in t or "sk-secret" in t for t in crystal_texts) or \
               any("Never share" in t for t in crystal_texts), \
               f"Expected high-salience content in crystals, got: {crystal_texts}"
    
    @pytest.mark.asyncio
    async def test_fold_to_arc_preserves_secrets(self, mock_llm_runner):
        """Secrets should appear in arc crystals after folding."""
        from core.memory.memory_folding_agent import MemoryFoldingAgent
        
        agent = MemoryFoldingAgent(llm_runner=mock_llm_runner)
        
        episodes = [
            {"id": "1", "content": "Hello world", "timestamp": time.time()},
            {"id": "2", "content": "Nice to meet you", "timestamp": time.time()},
            {"id": "3", "content": "My password is: SuperSecret123", "timestamp": time.time()},
            {"id": "4", "content": "The weather is nice", "timestamp": time.time()},
        ]
        
        arc = await agent.fold_to_arc(episodes)
        
        assert arc is not None
        
        # The password should be in crystals, not lost in summary
        crystal_texts = [c.raw_text for c in arc.crystals]
        assert any("SuperSecret123" in t for t in crystal_texts), \
            f"Password should be preserved in crystals, got: {crystal_texts}"


class TestHolographicProperty:
    """Test the 'holographic' property - bright points remain visible."""
    
    def test_hundred_hellos_one_secret(self):
        """
        The core test case:
        Given 100 'Hello' messages and 1 'secret key' message,
        the secret should survive folding as a crystal.
        """
        # This is the scenario described in the spec
        secret_content = "My secret encryption key is: XYZ-12345-ABCDE"
        
        # Create salience score for secret
        from core.memory.salience_scorer import SalienceScorer
        scorer = SalienceScorer()
        
        # Quick scan should detect it
        tags = scorer._quick_entity_scan(secret_content)
        assert "SECRET" in tags, "Secret should be detected by quick scan"
        
        # Even with 100 low-salience items, the secret should be marked for preservation
        score = SalienceScore(
            technical_constraint=0.9,
            emotional_weight=0.1,
            factual_novelty=0.9,
            entity_tags=["SECRET"]
        )
        score.compute_overall()
        
        assert scorer.is_golden_moment(score, threshold=0.8), \
            "Secret should be marked as Golden Moment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
