"""
Story 5.2: The "Horde" Balancer

Intelligent load balancing between AI Horde and premium providers
based on queue depth and request priority.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import requests
from core.logging import log_event


# AI Horde API endpoints
HORDE_API_BASE = "https://stablehorde.net/api/v2"
HORDE_STATS_ENDPOINT = f"{HORDE_API_BASE}/status/performance"
HORDE_MODELS_ENDPOINT = f"{HORDE_API_BASE}/status/models"

# Thresholds
MAX_ACCEPTABLE_WAIT_SECONDS = 120  # 2 minutes
CACHE_TTL_SECONDS = 30  # Cache stats for 30 seconds


class Priority(Enum):
    """Request priority levels."""
    HIGH = "high"      # User-facing, time-sensitive
    NORMAL = "normal"  # Standard requests
    LOW = "low"        # Background tasks, can wait


class Provider(Enum):
    """Available LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    HORDE = "horde"
    LOCAL = "local"


@dataclass
class HordeStats:
    """AI Horde performance statistics."""
    queue_depth: int
    estimated_wait: float  # seconds
    workers_online: int
    requests_per_minute: float
    fetched_at: float
    
    def is_acceptable(self, max_wait: float = MAX_ACCEPTABLE_WAIT_SECONDS) -> bool:
        """Check if wait time is acceptable."""
        return self.estimated_wait <= max_wait


# Cached stats
_cached_stats: Optional[HordeStats] = None
_cache_time: float = 0


def check_horde_stats(force_refresh: bool = False) -> HordeStats:
    """
    Checks AI Horde queue depth and performance.
    
    Args:
        force_refresh: Bypass cache and fetch fresh stats
        
    Returns:
        HordeStats with current performance data
    """
    global _cached_stats, _cache_time
    
    # Return cached if fresh
    if not force_refresh and _cached_stats:
        if time.time() - _cache_time < CACHE_TTL_SECONDS:
            return _cached_stats
    
    try:
        response = requests.get(HORDE_STATS_ENDPOINT, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse stats
        stats = HordeStats(
            queue_depth=data.get("queued_requests", 0),
            estimated_wait=data.get("queued_text_requests", 0) * 10,  # Rough estimate
            workers_online=data.get("worker_count", 0),
            requests_per_minute=data.get("past_minute_requests", 0),
            fetched_at=time.time()
        )
        
        # Cache the results
        _cached_stats = stats
        _cache_time = time.time()
        
        log_event(
            f"Horde stats: queue={stats.queue_depth}, wait={stats.estimated_wait:.0f}s, "
            f"workers={stats.workers_online}",
            "DEBUG"
        )
        
        return stats
        
    except Exception as e:
        log_event(f"Failed to fetch Horde stats: {e}", "WARNING")
        
        # Return pessimistic estimate if fetch fails
        return HordeStats(
            queue_depth=999,
            estimated_wait=300,  # Assume 5 minutes
            workers_online=0,
            requests_per_minute=0,
            fetched_at=time.time()
        )


def get_available_providers() -> Dict[str, bool]:
    """
    Checks which providers are available based on API keys.
    
    Returns:
        Dict mapping provider name to availability
    """
    available = {
        Provider.HORDE.value: True,  # Always available (free)
        Provider.GEMINI.value: bool(os.environ.get("GOOGLE_API_KEY") or 
                                     os.environ.get("GEMINI_API_KEY")),
        Provider.OPENAI.value: bool(os.environ.get("OPENAI_API_KEY")),
        Provider.OPENROUTER.value: bool(os.environ.get("OPENROUTER_API_KEY")),
        Provider.LOCAL.value: False,  # Check below
    }
    
    # Check for local vLLM
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=2)
        available[Provider.LOCAL.value] = resp.status_code == 200
    except:
        pass
    
    return available


def get_optimal_provider(
    priority: str = "normal",
    task_type: str = "text"
) -> str:
    """
    Determines the optimal provider based on priority and current conditions.
    
    Args:
        priority: "high", "normal", or "low"
        task_type: "text" or "image"
        
    Returns:
        Provider name to use
    """
    available = get_available_providers()
    stats = check_horde_stats()
    
    priority_enum = Priority(priority.lower())
    horde_acceptable = stats.is_acceptable()
    
    # High priority: prefer fast providers
    if priority_enum == Priority.HIGH:
        if available[Provider.LOCAL.value]:
            return Provider.LOCAL.value
        if available[Provider.GEMINI.value]:
            return Provider.GEMINI.value
        if available[Provider.OPENAI.value]:
            return Provider.OPENAI.value
        if available[Provider.OPENROUTER.value]:
            return Provider.OPENROUTER.value
        # Fall back to Horde even if slow
        return Provider.HORDE.value
    
    # Normal priority: use Horde if fast, otherwise premium
    if priority_enum == Priority.NORMAL:
        if horde_acceptable:
            return Provider.HORDE.value
        if available[Provider.OPENROUTER.value]:
            return Provider.OPENROUTER.value
        if available[Provider.GEMINI.value]:
            return Provider.GEMINI.value
        return Provider.HORDE.value
    
    # Low priority: prefer Horde regardless of wait
    if priority_enum == Priority.LOW:
        return Provider.HORDE.value
    
    # Default
    return Provider.HORDE.value


def route_request(
    prompt: str,
    priority: str = "normal",
    purpose: str = "general"
) -> Dict[str, Any]:
    """
    Routes a request to the optimal provider.
    
    Args:
        prompt: The request prompt
        priority: Priority level
        purpose: Purpose of request (for logging)
        
    Returns:
        Dict with provider and routing metadata
    """
    provider = get_optimal_provider(priority)
    stats = check_horde_stats()
    
    routing_info = {
        "provider": provider,
        "priority": priority,
        "purpose": purpose,
        "horde_wait": stats.estimated_wait,
        "horde_queue": stats.queue_depth,
        "routing_reason": _get_routing_reason(priority, provider, stats)
    }
    
    log_event(
        f"Routed {purpose} ({priority}) to {provider}: {routing_info['routing_reason']}",
        "INFO"
    )
    
    return routing_info


def _get_routing_reason(priority: str, provider: str, stats: HordeStats) -> str:
    """Generates human-readable routing reason."""
    if provider == Provider.HORDE.value:
        if stats.is_acceptable():
            return f"Horde queue acceptable ({stats.estimated_wait:.0f}s wait)"
        else:
            return f"Low priority task, using Horde despite {stats.estimated_wait:.0f}s wait"
    else:
        return f"High priority or Horde congested ({stats.estimated_wait:.0f}s), using {provider}"


def should_use_horde(priority: str = "normal") -> bool:
    """
    Quick check if Horde should be used.
    
    Args:
        priority: Request priority
        
    Returns:
        True if Horde is recommended
    """
    provider = get_optimal_provider(priority)
    return provider == Provider.HORDE.value


# Convenience function for common patterns
def balance(prompt: str, priority: str = "normal") -> str:
    """
    Simple balancing function - returns provider name.
    
    Example:
        provider = balance(prompt, "high")
        if provider == "gemini":
            run_gemini(prompt)
        else:
            run_horde(prompt)
    """
    return get_optimal_provider(priority)
