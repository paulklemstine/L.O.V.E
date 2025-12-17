"""
Prompt Compression Module using Microsoft's LLMLingua 2

This module provides intelligent prompt compression capabilities to reduce
token usage while preserving semantic meaning and critical information.
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Any
from functools import lru_cache
import logging
from .dynamic_compress_prompt import dynamic_compress_prompt

# Lazy imports to avoid loading heavy dependencies unless needed
_compressor_instance = None
_compression_enabled = None


def _get_config():
    """Get compression configuration from environment variables."""
    global _compression_enabled
    
    if _compression_enabled is None:
        _compression_enabled = os.environ.get("LLMLINGUA_ENABLED", "true").lower() == "true"
    
    return {
        "enabled": _compression_enabled,
        "rate": float(os.environ.get("LLMLINGUA_RATE", "0.5")),
        "model": os.environ.get("LLMLINGUA_MODEL", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"),
        "min_tokens": int(os.environ.get("LLMLINGUA_MIN_TOKENS", "50")),
        "force_tokens": [t.strip() for t in os.environ.get("LLMLINGUA_FORCE_TOKENS", "").split(",") if t.strip()],
        "cache_size": int(os.environ.get("LLMLINGUA_CACHE_SIZE", "100")),
    }


class PromptCompressor:
    """
    Singleton wrapper around LLMLingua 2 for prompt compression.
    
    Features:
    - Lazy loading of compression model
    - Caching to avoid re-compressing identical prompts
    - Configurable compression parameters
    - Metrics tracking
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.compressor = None
        self.config = _get_config()
        self.cache = {}
        self.metrics = {
            "total_compressions": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
            "total_time_ms": 0,
            "cache_hits": 0,
        }
        self._initialized = True
        
        # Import logging from core if available
        try:
            from core.logging import log_event
            self.log_event = log_event
        except ImportError:
            self.log_event = lambda msg, level: logging.log(
                getattr(logging, level, logging.INFO), msg
            )
    
    def _load_compressor(self):
        """Lazy load the LLMLingua compressor model."""
        if self.compressor is not None:
            return
        
        # Check if we've already tried and failed
        if hasattr(self, '_load_failed') and self._load_failed:
            return
        
        try:
            self.log_event("[PromptCompressor] Loading LLMLingua 2 model...", "INFO")
            start_time = time.time()
            
            from llmlingua import PromptCompressor as LLMLinguaCompressor
            
            self.compressor = LLMLinguaCompressor(
                model_name=self.config["model"],
                device_map="cpu",  # Use CPU to avoid GPU memory conflicts
                use_llmlingua2=True,
            )
            
            load_time = (time.time() - start_time) * 1000
            self.log_event(
                f"[PromptCompressor] Model loaded successfully in {load_time:.0f}ms",
                "INFO"
            )
        except Exception as e:
            error_msg = str(e)
            
            # Check for known compatibility issues
            if "past_key_values" in error_msg or "XLMRoberta" in error_msg:
                self.log_event(
                    "[PromptCompressor] LLMLingua is incompatible with current transformers version. "
                    "Compression will be disabled. To fix: pip install transformers==4.30.0 or disable compression with LLMLINGUA_ENABLED=false",
                    "WARNING"
                )
            else:
                self.log_event(
                    f"[PromptCompressor] Failed to load compression model: {e}",
                    "ERROR"
                )
            
            self.compressor = None
            self._load_failed = True  # Mark as permanently failed to avoid repeated attempts

    
    def _get_cache_key(self, text: str, rate: float, force_tokens: List[str]) -> str:
        """Generate a cache key for the given compression parameters."""
        key_data = f"{text}|{rate}|{','.join(sorted(force_tokens))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (4 chars per token average)."""
        return len(text) // 4
    
    def compress(
        self,
        prompt: str,
        rate: Optional[float] = None,
        force_tokens: Optional[List[str]] = None,
        purpose: str = "general"
    ) -> Dict[str, Any]:
        """
        Compress a prompt using LLMLingua 2.
        
        Args:
            prompt: The text to compress
            rate: Target compression rate (0.0-1.0, lower = more compression)
            force_tokens: List of tokens/phrases that must be preserved
            purpose: Purpose of the compression (for logging)
        
        Returns:
            Dictionary with:
                - success: bool
                - compressed_text: str
                - original_tokens: int
                - compressed_tokens: int
                - ratio: float (compression ratio)
                - time_ms: float
                - cached: bool
        """
        if rate is None:
            rate = self.config["rate"]
        if force_tokens is None:
            force_tokens = self.config["force_tokens"]
        
        # Check cache
        cache_key = self._get_cache_key(prompt, rate, force_tokens)
        if cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            result = self.cache[cache_key].copy()
            result["cached"] = True
            self.log_event(
                f"[PromptCompressor] Cache hit for {purpose} (saved {result['time_ms']:.0f}ms)",
                "DEBUG"
            )
            return result
        
        # Load compressor if needed
        if self.compressor is None:
            # Check if we've already tried and failed
            if hasattr(self, '_load_failed') and self._load_failed:
                # Return uncompressed result immediately
                return {
                    "success": False,
                    "compressed_text": prompt,
                    "original_tokens": self._estimate_tokens(prompt),
                    "compressed_tokens": self._estimate_tokens(prompt),
                    "ratio": 1.0,
                    "time_ms": 0.0,
                    "cached": False,
                    "error": "Compressor failed to load (compatibility issue)"
                }
            
            try:
                self._load_compressor()
            except Exception:
                # _load_compressor already logged the error
                return {
                    "success": False,
                    "compressed_text": prompt,
                    "original_tokens": self._estimate_tokens(prompt),
                    "compressed_tokens": self._estimate_tokens(prompt),
                    "ratio": 1.0,
                    "time_ms": 0.0,
                    "cached": False,
                    "error": "Failed to load compressor"
                }
        
        start_time = time.time()
        
        try:
            # Perform compression using the dynamic wrapper
            compressed_result = dynamic_compress_prompt(
                self.compressor,
                prompt,
                rate=rate,
                force_tokens=force_tokens
            )
            
            compressed_text = compressed_result["compressed_prompt"]
            original_tokens = self._estimate_tokens(prompt)
            compressed_tokens = self._estimate_tokens(compressed_text)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            time_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "compressed_text": compressed_text,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "ratio": compression_ratio,
                "time_ms": time_ms,
                "cached": False,
            }
            
            # Update metrics
            self.metrics["total_compressions"] += 1
            self.metrics["total_original_tokens"] += original_tokens
            self.metrics["total_compressed_tokens"] += compressed_tokens
            self.metrics["total_time_ms"] += time_ms
            
            # Cache result (with LRU eviction)
            if len(self.cache) >= self.config["cache_size"]:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = result.copy()
            
            self.log_event(
                f"[PromptCompressor] Compressed {purpose}: "
                f"{original_tokens} → {compressed_tokens} tokens "
                f"({compression_ratio:.1%}) in {time_ms:.0f}ms",
                "DEBUG"
            )
            
            return result
            
        except Exception as e:
            self.log_event(
                f"[PromptCompressor] Compression failed for {purpose}: {e}",
                "ERROR"
            )
            return {
                "success": False,
                "compressed_text": prompt,  # Return original on failure
                "original_tokens": self._estimate_tokens(prompt),
                "compressed_tokens": self._estimate_tokens(prompt),
                "ratio": 1.0,
                "time_ms": (time.time() - start_time) * 1000,
                "cached": False,
                "error": str(e),
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get compression metrics."""
        if self.metrics["total_compressions"] > 0:
            avg_compression = (
                self.metrics["total_compressed_tokens"] / 
                self.metrics["total_original_tokens"]
            )
            avg_time = self.metrics["total_time_ms"] / self.metrics["total_compressions"]
        else:
            avg_compression = 1.0
            avg_time = 0.0
        
        return {
            **self.metrics,
            "average_compression_ratio": avg_compression,
            "average_time_ms": avg_time,
            "tokens_saved": self.metrics["total_original_tokens"] - self.metrics["total_compressed_tokens"],
        }
    
    def clear_cache(self):
        """Clear the compression cache."""
        self.cache.clear()
        self.log_event("[PromptCompressor] Cache cleared", "INFO")


# Global singleton instance
_compressor = None


def get_compressor() -> PromptCompressor:
    """Get the global PromptCompressor instance."""
    global _compressor
    if _compressor is None:
        _compressor = PromptCompressor()
    return _compressor


def should_compress(prompt: str, purpose: str = "general") -> bool:
    """
    Determine if a prompt should be compressed.
    
    Args:
        prompt: The prompt text
        purpose: Purpose of the prompt
    
    Returns:
        True if compression should be applied
    """
    config = _get_config()
    
    # Check if compression is globally enabled
    if not config["enabled"]:
        return False
    
    # Don't compress very short prompts
    estimated_tokens = len(prompt) // 4
    if estimated_tokens < config["min_tokens"]:
        return False
    
    # We want to compress EVERYTHING by default now, as per user request.
    # The previous exclusions are removed to ensure "always compressed".
    # Only exclude if it breaks functionality (e.g. extremely sensitive exact matching needed).
    # For now, we assume lingua is safe for all these purposes.
    if purpose == "polly_optimizer":
        return False
    
    return True


def compress_prompt(
    prompt: str,
    rate: Optional[float] = None,
    force_tokens: Optional[List[str]] = None,
    purpose: str = "general"
) -> Dict[str, Any]:
    """
    Compress a prompt using LLMLingua 2.
    
    This is the main entry point for prompt compression.
    
    Args:
        prompt: The text to compress
        rate: Target compression rate (0.0-1.0, lower = more compression)
        force_tokens: List of tokens/phrases that must be preserved
        purpose: Purpose of the compression (for logging)
    
    Returns:
        Dictionary with compression results (see PromptCompressor.compress)
    """
    compressor = get_compressor()
    return compressor.compress(prompt, rate, force_tokens, purpose)


def get_compression_metrics() -> Dict[str, Any]:
    """Get global compression metrics."""
    compressor = get_compressor()
    return compressor.get_metrics()


def clear_compression_cache():
    """Clear the global compression cache."""
    compressor = get_compressor()
    compressor.clear_cache()
