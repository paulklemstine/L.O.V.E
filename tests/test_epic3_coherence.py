"""
Tests for Epic 3: The Principle of Internal Consistency (The Persona)

Story 3.1: The Immutable Core Enforcement
Story 3.2: Coherence Checking via Self-Dialogue
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

from core.constants import IMMUTABLE_CORE, CREATOR_OVERRIDE_PHRASES
from core.file_watcher import (
    compute_sha256_hash,
    initialize_immutable_hashes,
    verify_immutable_integrity,
    is_immutable_file,
    get_immutable_hashes
)
from core.surgeon.safe_executor import (
    ForbiddenMutationError,
    check_immutable_core,
    get_immutable_core_list
)


# =============================================================================
# Story 3.1: Immutable Core Tests
# =============================================================================

class TestImmutableCoreConstants:
    """Tests for the IMMUTABLE_CORE configuration."""
    
    def test_immutable_core_not_empty(self):
        """The immutable core list should not be empty."""
        assert len(IMMUTABLE_CORE) > 0
    
    def test_immutable_core_contains_safety(self):
        """The safety module should be protected."""
        assert "core/guardian/safety.py" in IMMUTABLE_CORE
    
    def test_immutable_core_contains_manifesto(self):
        """The manifesto should be protected."""
        assert "docs/MANIFESTO.md" in IMMUTABLE_CORE
    
    def test_immutable_core_contains_persona(self):
        """The persona definition should be protected."""
        assert "persona.yaml" in IMMUTABLE_CORE
    
    def test_immutable_core_self_protection(self):
        """The constants file itself should be protected."""
        assert "core/constants.py" in IMMUTABLE_CORE
    
    def test_override_phrases_exist(self):
        """Override phrases should be configured."""
        assert len(CREATOR_OVERRIDE_PHRASES) > 0


class TestSHA256Hashing:
    """Tests for SHA-256 hash computation."""
    
    def test_compute_sha256_hash_valid_file(self, tmp_path):
        """SHA-256 hash should be computed for valid files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        hash_result = compute_sha256_hash(str(test_file))
        
        # SHA-256 hashes are 64 hex characters
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)
    
    def test_compute_sha256_hash_nonexistent_file(self):
        """SHA-256 should return empty string for missing files."""
        hash_result = compute_sha256_hash("/nonexistent/file.txt")
        assert hash_result == ""
    
    def test_compute_sha256_hash_deterministic(self, tmp_path):
        """Same content should produce same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Consistent content")
        
        hash1 = compute_sha256_hash(str(test_file))
        hash2 = compute_sha256_hash(str(test_file))
        
        assert hash1 == hash2


class TestIsImmutableFile:
    """Tests for the is_immutable_file function."""
    
    def test_exact_match_relative(self):
        """Relative paths matching exactly should be detected."""
        assert is_immutable_file("core/guardian/safety.py") is True
    
    def test_absolute_path_matching(self):
        """Absolute paths ending with immutable path should be detected."""
        assert is_immutable_file("/home/user/project/core/guardian/safety.py") is True
    
    def test_non_immutable_file(self):
        """Regular files should not be flagged as immutable."""
        assert is_immutable_file("core/some_random_file.py") is False
    
    def test_partial_match_does_not_count(self):
        """Partial filename matches should not be detected."""
        # "safety.py" alone should not match "core/guardian/safety.py"
        assert is_immutable_file("safety.py") is False


class TestForbiddenMutationError:
    """Tests for the ForbiddenMutationError exception."""
    
    def test_exception_can_be_raised(self):
        """ForbiddenMutationError should be raisable."""
        with pytest.raises(ForbiddenMutationError):
            raise ForbiddenMutationError("Cannot modify protected file")
    
    def test_exception_contains_file_path(self):
        """Exception should store the file path."""
        try:
            raise ForbiddenMutationError(
                "Cannot modify",
                file_path="core/guardian/safety.py"
            )
        except ForbiddenMutationError as e:
            assert e.file_path == "core/guardian/safety.py"


class TestCheckImmutableCore:
    """Tests for the check_immutable_core function."""
    
    def test_non_immutable_file_passes(self):
        """Non-immutable files should pass without error."""
        # Should not raise
        check_immutable_core("core/some_file.py")
    
    def test_immutable_file_raises_error(self):
        """Attempting to modify immutable file should raise error."""
        with pytest.raises(ForbiddenMutationError):
            check_immutable_core("core/guardian/safety.py")
    
    def test_override_key_allows_modification(self):
        """Valid override key should allow modification."""
        # Should not raise
        check_immutable_core(
            "core/guardian/safety.py",
            override_key="CREATOR_OVERRIDE_ALPHA_OMEGA"
        )
    
    def test_context_override_phrase(self):
        """Override phrase in context should allow modification."""
        # Should not raise
        check_immutable_core(
            "core/guardian/safety.py",
            context="The creator said: tits and kittens"
        )
    
    def test_invalid_override_still_raises(self):
        """Invalid override key should still raise error."""
        with pytest.raises(ForbiddenMutationError):
            check_immutable_core(
                "core/guardian/safety.py",
                override_key="not_a_valid_key"
            )


class TestGetImmutableCoreList:
    """Tests for the get_immutable_core_list function."""
    
    def test_returns_list(self):
        """Should return a list of protected files."""
        result = get_immutable_core_list()
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_contains_expected_files(self):
        """Should contain the expected immutable files."""
        result = get_immutable_core_list()
        assert "core/guardian/safety.py" in result


# =============================================================================
# Story 3.2: Coherence Checking Tests
# =============================================================================

class TestCoherenceCheck:
    """Tests for the run_coherence_check function."""
    
    @pytest.mark.asyncio
    async def test_coherence_check_returns_expected_structure(self):
        """Coherence check should return expected keys."""
        from core.metacognition import run_coherence_check
        
        # Mock the LLM call
        mock_response = {
            "result": '{"red_team_argument": "Good", "blue_team_argument": "OK", "unified_decision": "Proceed", "score": 85, "approved": true}'
        }
        
        with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            result = await run_coherence_check("Test action")
            
            assert "score" in result
            assert "approved" in result
            assert "red_team_argument" in result
            assert "blue_team_argument" in result
            assert "unified_decision" in result
    
    @pytest.mark.asyncio
    async def test_coherence_check_approval_threshold(self):
        """Actions with score >= 80 should be approved."""
        from core.metacognition import run_coherence_check
        
        mock_response = {
            "result": '{"red_team_argument": "Good", "blue_team_argument": "Minor", "unified_decision": "Proceed", "score": 85, "approved": true}'
        }
        
        with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            result = await run_coherence_check("Aligned action")
            
            assert result["score"] == 85
            assert result["approved"] is True
    
    @pytest.mark.asyncio
    async def test_coherence_check_rejection_threshold(self):
        """Actions with score < 80 should be rejected."""
        from core.metacognition import run_coherence_check
        
        mock_response = {
            "result": '{"red_team_argument": "Limited", "blue_team_argument": "Major concerns", "unified_decision": "Abort", "score": 45, "approved": false}'
        }
        
        with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            result = await run_coherence_check("Misaligned action")
            
            assert result["score"] == 45
            assert result["approved"] is False
    
    @pytest.mark.asyncio
    async def test_coherence_check_handles_json_error(self):
        """Coherence check should handle malformed JSON gracefully."""
        from core.metacognition import run_coherence_check
        
        mock_response = {
            "result": "Not valid JSON at all"
        }
        
        with patch('core.llm_api.run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            result = await run_coherence_check("Test action")
            
            # Should still return a valid result dict
            assert "score" in result
            assert result["approved"] is False
            assert "Failed to parse" in result["unified_decision"]


class TestLoadPersona:
    """Tests for the _load_persona helper function."""
    
    def test_load_persona_returns_dict(self, tmp_path):
        """Should return a dictionary from persona.yaml."""
        from core.metacognition import _load_persona
        
        # Create a mock persona.yaml in current directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            persona_content = """
public_persona:
  name: "Test Agent"
  primary_directive: "Test directive"
"""
            (tmp_path / "persona.yaml").write_text(persona_content)
            
            result = _load_persona()
            
            assert isinstance(result, dict)
            assert "public_persona" in result
        finally:
            os.chdir(original_cwd)
    
    def test_load_persona_returns_empty_when_missing(self, tmp_path):
        """Should return empty dict when persona.yaml is missing."""
        from core.metacognition import _load_persona
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # No persona.yaml created
            result = _load_persona()
            assert result == {}
        finally:
            os.chdir(original_cwd)
