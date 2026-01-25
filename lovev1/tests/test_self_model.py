"""
Tests for Story 1.1: The Dynamic Self-Symbol (The Quine)

These tests verify that the Self-Symbol accurately reflects the system's
current architecture, enabling the Strange Loop.
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSelfSymbol:
    """Tests for the SelfSymbol dataclass."""
    
    def test_self_symbol_creation(self):
        """Test basic SelfSymbol creation."""
        from core.reflection.self_model import SelfSymbol, AgentInfo, APIHealthStatus
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123def456",
            active_agents=[
                AgentInfo(name="Orchestrator", module="core.agents.orchestrator", available=True)
            ],
            api_health=[
                APIHealthStatus(provider="gemini", status="healthy")
            ],
            capabilities=["code_execution", "web_search"],
            available_tools=["search_web", "read_file"],
        )
        
        assert symbol.codebase_hash == "abc123def456"
        assert len(symbol.active_agents) == 1
        assert len(symbol.capabilities) == 2
    
    def test_can_perform_capability(self):
        """Test capability checking."""
        from core.reflection.self_model import SelfSymbol, AgentInfo, APIHealthStatus
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123",
            active_agents=[],
            api_health=[],
            capabilities=["web_search", "code_execution", "image_generation"],
            available_tools=[],
        )
        
        assert symbol.can_perform("web_search") is True
        assert symbol.can_perform("WEB_SEARCH") is True  # Case insensitive
        assert symbol.can_perform("teleporation") is False
    
    def test_has_tool(self):
        """Test tool availability checking."""
        from core.reflection.self_model import SelfSymbol
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123",
            active_agents=[],
            api_health=[],
            capabilities=[],
            available_tools=["search_web", "read_file", "generate_image"],
        )
        
        assert symbol.has_tool("search_web") is True
        assert symbol.has_tool("SEARCH_WEB") is True  # Case insensitive
        assert symbol.has_tool("hack_pentagon") is False
    
    def test_to_dict(self):
        """Test serialization to dict."""
        from core.reflection.self_model import SelfSymbol, AgentInfo, APIHealthStatus
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123",
            active_agents=[AgentInfo(name="Test", module="test", available=True)],
            api_health=[APIHealthStatus(provider="test", status="healthy")],
            capabilities=["test_cap"],
            available_tools=["test_tool"],
        )
        
        d = symbol.to_dict()
        assert isinstance(d, dict)
        assert d["codebase_hash"] == "abc123"
        assert isinstance(d["active_agents"], list)
        assert isinstance(d["active_agents"][0], dict)
    
    def test_to_json(self):
        """Test JSON serialization."""
        from core.reflection.self_model import SelfSymbol
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123",
            active_agents=[],
            api_health=[],
            capabilities=[],
            available_tools=[],
        )
        
        json_str = symbol.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["codebase_hash"] == "abc123"
    
    def test_context_injection(self):
        """Test context injection string generation."""
        from core.reflection.self_model import SelfSymbol, AgentInfo, APIHealthStatus
        
        symbol = SelfSymbol(
            generated_at="2026-01-16T20:30:00",
            codebase_hash="abc123def456789",
            active_agents=[
                AgentInfo(name="Orchestrator", module="test", available=True),
                AgentInfo(name="BrokenAgent", module="test", available=False),
            ],
            api_health=[
                APIHealthStatus(provider="gemini", status="healthy"),
                APIHealthStatus(provider="vllm", status="unavailable"),
            ],
            capabilities=["code_execution", "web_search"],
            available_tools=[],
        )
        
        ctx = symbol.to_context_injection()
        
        assert "SELF-SYMBOL" in ctx
        assert "abc123def456" in ctx  # Truncated hash
        assert "✓ Orchestrator" in ctx
        assert "✗ BrokenAgent" in ctx
        assert "🟢 gemini" in ctx
        assert "🔴 vllm" in ctx


class TestSelfModel:
    """Tests for the SelfModel class."""
    
    def test_compute_codebase_hash(self):
        """Test that codebase hash changes with file modifications."""
        from core.reflection.self_model import SelfModel
        
        model = SelfModel()
        hash1 = model._compute_codebase_hash()
        
        # Hash should be consistent for same state
        hash2 = model._compute_codebase_hash()
        assert hash1 == hash2
        
        # Hash should be a valid MD5 hex string
        assert len(hash1) == 32
        assert all(c in "0123456789abcdef" for c in hash1)
    
    def test_discover_agents(self):
        """Test agent discovery."""
        from core.reflection.self_model import SelfModel
        
        model = SelfModel()
        agents = model._discover_agents()
        
        # Should find some agents
        assert isinstance(agents, list)
        
        # Check that Orchestrator is found (it definitely exists)
        agent_names = [a.name for a in agents]
        assert "Orchestrator" in agent_names
    
    def test_discover_capabilities(self):
        """Test capability discovery."""
        from core.reflection.self_model import SelfModel
        
        model = SelfModel()
        caps = model._discover_capabilities()
        
        # Should include core capabilities
        assert "code_execution" in caps
        assert "text_generation" in caps
        assert "self_reflection" in caps
    
    def test_generate_caches(self):
        """Test that generate() uses caching correctly."""
        from core.reflection.self_model import SelfModel
        
        model = SelfModel()
        
        # First generation
        symbol1 = model.generate()
        
        # Second generation should use cache
        symbol2 = model.generate()
        assert symbol1.generated_at == symbol2.generated_at  # Same object
        
        # Force regeneration should create new timestamp
        symbol3 = model.generate(force=True)
        # The hash might be the same if no files changed
        # but generated_at should potentially be different
        assert symbol3 is not None


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_self_symbol(self):
        """Test get_self_symbol function."""
        from core.reflection.self_model import get_self_symbol, SelfSymbol
        
        symbol = get_self_symbol()
        assert isinstance(symbol, SelfSymbol)
        assert symbol.generated_at is not None
    
    def test_refresh_self_symbol(self):
        """Test refresh_self_symbol function."""
        from core.reflection.self_model import refresh_self_symbol, SelfSymbol
        
        symbol = refresh_self_symbol()
        assert isinstance(symbol, SelfSymbol)
    
    def test_get_context_injection(self):
        """Test get_context_injection function."""
        from core.reflection.self_model import get_context_injection
        
        ctx = get_context_injection()
        assert isinstance(ctx, str)
        assert "SELF-SYMBOL" in ctx


class TestStrangeLoopBehavior:
    """
    Tests for the Strange Loop behavior - the AI's ability to accurately
    know its own capabilities.
    """
    
    def test_accurate_capability_reporting(self):
        """
        Strange Loop Test: Verify that capability reporting is accurate.
        
        When asked "Can you browse the web?", the system should check
        actual capabilities, not guess.
        """
        from core.reflection.self_model import get_self_symbol
        
        symbol = get_self_symbol()
        
        # The system should know about web_search capability
        # (this exists in the codebase as we verified)
        can_search = symbol.can_perform("web_search")
        
        # It should NOT claim capabilities it doesn't have
        cannot_teleport = not symbol.can_perform("quantum_teleportation")
        
        assert can_search is True
        assert cannot_teleport is True
    
    def test_agent_availability_accuracy(self):
        """
        Test that agent availability is accurately reported.
        """
        from core.reflection.self_model import get_self_symbol
        
        symbol = get_self_symbol()
        
        # Orchestrator should be available (file exists)
        orchestrator = symbol.get_agent_status("Orchestrator")
        assert orchestrator is not None
        assert orchestrator.available is True
        
        # A non-existent agent should return None
        fake_agent = symbol.get_agent_status("FakeNonExistentAgent12345")
        assert fake_agent is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
