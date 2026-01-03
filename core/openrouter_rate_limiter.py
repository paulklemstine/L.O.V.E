"""
OpenRouter Rate Limiter

Precision tracking and rate limiting for OpenRouter API calls.
Enforces a configurable limit (default 999 calls) per 24-hour rolling window.
"""

import os
import json
import time
import threading
from typing import Optional
from core.logging import log_event

# --- Configuration ---
OPENROUTER_RATE_LIMIT = 999
OPENROUTER_WINDOW_SECONDS = 24 * 60 * 60  # 24 hours


class OpenRouterRateLimiter:
    """
    Thread-safe rate limiter for OpenRouter API calls.
    
    Uses a rolling 24-hour window to track calls and enforce limits.
    Persists call history to disk for survival across restarts.
    """
    
    def __init__(self, limit: int = OPENROUTER_RATE_LIMIT, 
                 window_seconds: int = OPENROUTER_WINDOW_SECONDS,
                 storage_path: Optional[str] = None):
        """
        Initialize the rate limiter.
        
        Args:
            limit: Maximum number of calls allowed in the window (default: 999)
            window_seconds: Size of the rolling window in seconds (default: 24 hours)
            storage_path: Path to store call history JSON (default: project root)
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.RLock()
        
        # Determine storage path
        if storage_path is None:
            # Store in the project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_path = os.path.join(project_root, "openrouter_calls.json")
        self.storage_path = storage_path
        
        # Call timestamps (Unix timestamps as floats)
        self.call_timestamps: list[float] = []
        
        # Load existing state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load call history from persistent storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.call_timestamps = data.get("call_timestamps", [])
                    # Prune old calls on load
                    self._prune_old_calls()
                    log_event(f"OpenRouter rate limiter: Loaded {len(self.call_timestamps)} calls from storage.", "INFO")
        except (json.JSONDecodeError, IOError) as e:
            log_event(f"OpenRouter rate limiter: Could not load state: {e}", "WARNING")
            self.call_timestamps = []
    
    def _save_state(self) -> None:
        """Save call history to persistent storage."""
        try:
            data = {
                "call_timestamps": self.call_timestamps,
                "limit": self.limit,
                "window_seconds": self.window_seconds,
                "last_updated": time.time()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            log_event(f"OpenRouter rate limiter: Could not save state: {e}", "ERROR")
    
    def _prune_old_calls(self) -> None:
        """Remove calls older than the rolling window."""
        cutoff = time.time() - self.window_seconds
        self.call_timestamps = [ts for ts in self.call_timestamps if ts > cutoff]
    
    def record_call(self) -> None:
        """
        Record a new API call.
        
        Should be called after a successful OpenRouter API request.
        """
        with self._lock:
            now = time.time()
            self.call_timestamps.append(now)
            self._prune_old_calls()
            self._save_state()
            
            remaining = self.get_remaining_calls()
            if remaining <= 100:
                log_event(f"OpenRouter rate limiter: WARNING - Only {remaining} calls remaining in 24h window!", "WARNING")
            elif remaining <= 50:
                log_event(f"OpenRouter rate limiter: CRITICAL - Only {remaining} calls remaining!", "CRITICAL")
    
    def get_remaining_calls(self) -> int:
        """
        Get the number of remaining calls in the current window.
        
        Returns:
            Number of calls remaining before hitting the limit.
        """
        with self._lock:
            self._prune_old_calls()
            return max(0, self.limit - len(self.call_timestamps))
    
    def get_calls_made(self) -> int:
        """
        Get the number of calls made in the current window.
        
        Returns:
            Number of calls made in the current 24-hour window.
        """
        with self._lock:
            self._prune_old_calls()
            return len(self.call_timestamps)
    
    def is_rate_limited(self) -> bool:
        """
        Check if the rate limit has been reached.
        
        Returns:
            True if rate limit is reached, False otherwise.
        """
        return self.get_remaining_calls() <= 0
    
    def get_rate_limit_info(self) -> dict:
        """
        Get detailed rate limit status information.
        
        Returns:
            Dictionary with rate limit details including:
            - calls_made: Number of calls in current window
            - calls_remaining: Remaining calls before limit
            - limit: The configured limit
            - window_seconds: The window duration
            - is_limited: Whether limit is currently reached
            - oldest_call_age: Age of oldest call in window (seconds)
            - next_slot_available: Seconds until a call slot frees up (if limited)
        """
        with self._lock:
            self._prune_old_calls()
            
            calls_made = len(self.call_timestamps)
            calls_remaining = max(0, self.limit - calls_made)
            is_limited = calls_remaining <= 0
            
            oldest_call_age = None
            next_slot_available = None
            
            if self.call_timestamps:
                oldest_ts = min(self.call_timestamps)
                oldest_call_age = time.time() - oldest_ts
                
                if is_limited:
                    # Calculate when the oldest call will expire
                    next_slot_available = self.window_seconds - oldest_call_age
                    if next_slot_available < 0:
                        next_slot_available = 0
            
            return {
                "calls_made": calls_made,
                "calls_remaining": calls_remaining,
                "limit": self.limit,
                "window_seconds": self.window_seconds,
                "is_limited": is_limited,
                "oldest_call_age": oldest_call_age,
                "next_slot_available": next_slot_available,
                "storage_path": self.storage_path
            }
    
    def reset(self) -> None:
        """Reset the rate limiter (clears all recorded calls)."""
        with self._lock:
            self.call_timestamps = []
            self._save_state()
            log_event("OpenRouter rate limiter: Reset - all calls cleared.", "INFO")


# --- Singleton Instance ---
_rate_limiter_instance: Optional[OpenRouterRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_openrouter_rate_limiter() -> OpenRouterRateLimiter:
    """
    Get the singleton OpenRouter rate limiter instance.
    
    Returns:
        The global OpenRouterRateLimiter instance.
    """
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        with _rate_limiter_lock:
            # Double-check locking
            if _rate_limiter_instance is None:
                _rate_limiter_instance = OpenRouterRateLimiter()
    
    return _rate_limiter_instance
