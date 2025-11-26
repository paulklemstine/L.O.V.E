"""
Tests for the prompt compression module (LLMLingua 2 integration)
"""

import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prompt_compressor import (
    PromptCompressor,
    compress_prompt,
    should_compress,
    get_compression_metrics,
    clear_compression_cache,
    get_compressor,
)


class TestPromptCompressor:
    """Test the PromptCompressor class"""
    
    def test_singleton_pattern(self):
        """Test that PromptCompressor follows singleton pattern"""
        compressor1 = PromptCompressor()
        compressor2 = PromptCompressor()
        assert compressor1 is compressor2, "PromptCompressor should be a singleton"
    
    def test_get_compressor(self):
        """Test the get_compressor() function"""
        compressor = get_compressor()
        assert isinstance(compressor, PromptCompressor)
        assert compressor is get_compressor(), "Should return same instance"
    
    def test_estimate_tokens(self):
        """Test token estimation"""
        compressor = get_compressor()
        text = "This is a test prompt with some content."
        tokens = compressor._estimate_tokens(text)
        assert tokens > 0, "Should estimate non-zero tokens"
        assert tokens == len(text) // 4, "Should use 4 chars per token estimate"
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        compressor = get_compressor()
        text = "Test prompt"
        rate = 0.5
        force_tokens = ["test", "prompt"]
        
        key1 = compressor._get_cache_key(text, rate, force_tokens)
        key2 = compressor._get_cache_key(text, rate, force_tokens)
        
        assert key1 == key2, "Same inputs should generate same cache key"
        
        # Different inputs should generate different keys
        key3 = compressor._get_cache_key(text, 0.3, force_tokens)
        assert key1 != key3, "Different rate should generate different cache key"


class TestShouldCompress:
    """Test the should_compress() decision logic"""
    
    def test_short_prompt_skipped(self):
        """Test that very short prompts are not compressed"""
        short_prompt = "Short prompt"
        assert not should_compress(short_prompt), "Short prompts should not be compressed"
    
    def test_long_prompt_compressed(self):
        """Test that long prompts are compressed"""
        # Create a prompt longer than min_tokens (default 500 tokens = ~2000 chars)
        long_prompt = "This is a test prompt. " * 200  # ~4600 chars
        
        # Set compression enabled for this test
        os.environ["LLMLINGUA_ENABLED"] = "true"
        
        assert should_compress(long_prompt), "Long prompts should be compressed"
    
    def test_disabled_compression(self):
        """Test that compression can be disabled via environment variable"""
        os.environ["LLMLINGUA_ENABLED"] = "false"
        long_prompt = "This is a test prompt. " * 200
        
        assert not should_compress(long_prompt), "Compression should be disabled"
        
        # Re-enable for other tests
        os.environ["LLMLINGUA_ENABLED"] = "true"
    
    def test_skip_purposes(self):
        """Test that certain purposes skip compression"""
        long_prompt = "This is a test prompt. " * 200
        
        assert not should_compress(long_prompt, purpose="json_repair"), \
            "json_repair should skip compression"
        assert not should_compress(long_prompt, purpose="emotion"), \
            "emotion should skip compression"
        assert not should_compress(long_prompt, purpose="log_squash"), \
            "log_squash should skip compression"


class TestCompressPrompt:
    """Test the compress_prompt() function"""
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_COMPRESSION_TESTS") == "true",
        reason="Compression tests skipped (requires LLMLingua model download)"
    )
    def test_basic_compression(self):
        """Test basic prompt compression"""
        # Create a long prompt
        prompt = """
        You are a helpful AI assistant. Your task is to help users with their questions.
        You should always be polite, accurate, and helpful. When answering questions,
        make sure to provide detailed explanations and examples where appropriate.
        """ * 50  # Make it long enough to compress
        
        result = compress_prompt(prompt, purpose="test")
        
        assert result["success"], "Compression should succeed"
        assert "compressed_text" in result
        assert "original_tokens" in result
        assert "compressed_tokens" in result
        assert "ratio" in result
        assert "time_ms" in result
        
        # Verify compression actually reduced size
        assert result["compressed_tokens"] < result["original_tokens"], \
            "Compression should reduce token count"
        assert result["ratio"] < 1.0, "Compression ratio should be less than 1.0"
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_COMPRESSION_TESTS") == "true",
        reason="Compression tests skipped"
    )
    def test_force_tokens_preserved(self):
        """Test that force_tokens are preserved in compression"""
        prompt = """
        You have access to the following tools: get_weather, search_web, calculate.
        Use the tool_name and arguments to call these tools.
        """ * 20
        
        force_tokens = ["get_weather", "search_web", "calculate", "tool_name", "arguments"]
        
        result = compress_prompt(prompt, force_tokens=force_tokens, purpose="test")
        
        if result["success"]:
            compressed_text = result["compressed_text"]
            # Check that critical tokens are still present
            for token in force_tokens:
                assert token in compressed_text, f"Force token '{token}' should be preserved"
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_COMPRESSION_TESTS") == "true",
        reason="Compression tests skipped"
    )
    def test_compression_caching(self):
        """Test that compression results are cached"""
        prompt = "This is a test prompt for caching. " * 100
        
        # Clear cache first
        clear_compression_cache()
        
        # First call - should not be cached
        result1 = compress_prompt(prompt, purpose="test")
        assert not result1.get("cached", False), "First call should not be cached"
        
        # Second call with same prompt - should be cached
        result2 = compress_prompt(prompt, purpose="test")
        assert result2.get("cached", False), "Second call should be cached"
        
        # Results should be identical
        assert result1["compressed_text"] == result2["compressed_text"]
    
    def test_compression_fallback_on_error(self):
        """Test that compression gracefully falls back to original on error"""
        # This test doesn't require the actual model
        prompt = "Short test"
        
        result = compress_prompt(prompt, purpose="test")
        
        # Even if compression fails, should return original prompt
        assert "compressed_text" in result
        assert result["compressed_text"] == prompt or result["success"]


class TestMetrics:
    """Test compression metrics tracking"""
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_COMPRESSION_TESTS") == "true",
        reason="Compression tests skipped"
    )
    def test_metrics_tracking(self):
        """Test that compression metrics are tracked"""
        # Clear cache and metrics
        clear_compression_cache()
        compressor = get_compressor()
        compressor.metrics = {
            "total_compressions": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
            "total_time_ms": 0,
            "cache_hits": 0,
        }
        
        prompt = "This is a test prompt for metrics tracking. " * 100
        
        # Perform compression
        result = compress_prompt(prompt, purpose="test")
        
        if result["success"]:
            metrics = get_compression_metrics()
            
            assert metrics["total_compressions"] > 0, "Should track compressions"
            assert metrics["total_original_tokens"] > 0, "Should track original tokens"
            assert metrics["total_compressed_tokens"] >= 0, "Should track compressed tokens"
            assert metrics["total_time_ms"] > 0, "Should track time spent"
            assert "average_compression_ratio" in metrics
            assert "average_time_ms" in metrics
            assert "tokens_saved" in metrics


class TestIntegration:
    """Integration tests with actual LLM components"""
    
    def test_deepagent_integration(self):
        """Test that DeepAgentEngine can use compression"""
        # This is a smoke test - just verify imports work
        try:
            from core.deep_agent_engine import DeepAgentEngine
            # If we can import, the integration is at least syntactically correct
            assert True
        except ImportError as e:
            pytest.fail(f"DeepAgentEngine import failed: {e}")
    
    def test_gemini_react_integration(self):
        """Test that GeminiReActEngine can use compression"""
        try:
            from core.gemini_react_engine import GeminiReActEngine
            assert True
        except ImportError as e:
            pytest.fail(f"GeminiReActEngine import failed: {e}")
    
    def test_llm_api_integration(self):
        """Test that llm_api can use compression"""
        try:
            from core.llm_api import run_llm
            assert True
        except ImportError as e:
            pytest.fail(f"llm_api import failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
