"""
Story 5.1: Resource Scout

Discovers free API tiers and computation resources,
creating TODO items for evaluation and integration.
"""
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.logging import log_event


TODO_PATH = "TODO.md"

# Known API sources to check
KNOWN_API_SOURCES = [
    {
        "name": "AI Horde",
        "url": "https://stablehorde.net",
        "type": "image_and_text",
        "free_tier": True,
        "limits": "Queue-based, no hard limits"
    },
    {
        "name": "OpenRouter",
        "url": "https://openrouter.ai",
        "type": "llm",
        "free_tier": True,
        "limits": "Free models with rate limits"
    },
    {
        "name": "Pollinations.ai",
        "url": "https://pollinations.ai",
        "type": "image",
        "free_tier": True,
        "limits": "Unlimited free generation"
    },
    {
        "name": "Groq",
        "url": "https://groq.com",
        "type": "llm",
        "free_tier": True,
        "limits": "Free tier with quota"
    },
    {
        "name": "Together AI",
        "url": "https://together.ai",
        "type": "llm",
        "free_tier": True,
        "limits": "$25 free credits"
    },
    {
        "name": "Replicate",
        "url": "https://replicate.com",
        "type": "multi",
        "free_tier": True,
        "limits": "Free tier for open models"
    },
    {
        "name": "Hugging Face Inference",
        "url": "https://huggingface.co/inference-api",
        "type": "llm",
        "free_tier": True,
        "limits": "Free API for hosted models"
    },
]

# Search queries for discovering new APIs
DISCOVERY_QUERIES = [
    "new LLM API free tier 2024",
    "free AI image generation API",
    "open source LLM hosting free",
    "free GPU compute API",
    "AI API free credits",
]


@dataclass
class APIResource:
    """Represents a discovered API resource."""
    name: str
    url: str
    resource_type: str  # llm, image, compute, multi
    free_tier: bool
    limits: str
    discovered_at: str
    evaluated: bool = False
    notes: str = ""


def get_known_resources() -> List[APIResource]:
    """
    Returns list of known API resources.
    
    Returns:
        List of APIResource objects
    """
    resources = []
    for source in KNOWN_API_SOURCES:
        resources.append(APIResource(
            name=source["name"],
            url=source["url"],
            resource_type=source["type"],
            free_tier=source["free_tier"],
            limits=source["limits"],
            discovered_at="built-in"
        ))
    return resources


def search_for_new_apis(query: str = None) -> List[Dict[str, Any]]:
    """
    Searches for new API resources.
    
    This is a framework for future web search integration.
    Currently returns empty list - to be enhanced with actual search.
    
    Args:
        query: Search query string
        
    Returns:
        List of discovered API info dicts
    """
    if query is None:
        query = DISCOVERY_QUERIES[0]
    
    log_event(f"Searching for resources: {query}", "INFO")
    
    # Placeholder for actual web search integration
    # In production, this would use a search API or scraping
    discovered = []
    
    # For now, return empty - search integration to be added
    log_event(f"Search returned {len(discovered)} results", "INFO")
    return discovered


def extract_api_limits(url: str) -> Optional[Dict[str, Any]]:
    """
    Extracts API limit information from documentation.
    
    Framework for future documentation parsing.
    
    Args:
        url: URL of API documentation
        
    Returns:
        Dict with rate limits, quotas, etc.
    """
    log_event(f"Extracting limits from: {url}", "INFO")
    
    # Placeholder structure
    limits = {
        "rate_limit": None,
        "daily_quota": None,
        "free_credits": None,
        "restrictions": [],
    }
    
    # Documentation parsing to be implemented
    return limits


def create_evaluation_todo(api_info: APIResource) -> bool:
    """
    Creates a TODO item for evaluating an API.
    
    Args:
        api_info: APIResource to create TODO for
        
    Returns:
        True if TODO was created
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        todo_entry = f"\n## 🔍 API Evaluation: {api_info.name}\n"
        todo_entry += f"- [ ] Evaluate {api_info.name} integration\n"
        todo_entry += f"  - URL: {api_info.url}\n"
        todo_entry += f"  - Type: {api_info.resource_type}\n"
        todo_entry += f"  - Limits: {api_info.limits}\n"
        todo_entry += f"  - Added: {timestamp}\n"
        
        # Check if TODO.md exists
        if os.path.exists(TODO_PATH):
            with open(TODO_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Check if already in TODO
            if api_info.name in content:
                log_event(f"{api_info.name} already in TODO.md", "DEBUG")
                return False
        else:
            content = "# TODO\n\n"
        
        # Append new entry
        with open(TODO_PATH, "a", encoding="utf-8") as f:
            f.write(todo_entry)
        
        log_event(f"Created TODO for: {api_info.name}", "INFO")
        return True
        
    except Exception as e:
        log_event(f"Failed to create TODO: {e}", "ERROR")
        return False


def run_resource_scan() -> List[APIResource]:
    """
    Runs a complete resource scan workflow.
    
    Returns:
        List of all discovered resources
    """
    log_event("Starting resource scan...", "INFO")
    
    all_resources = []
    
    # Get known resources
    known = get_known_resources()
    all_resources.extend(known)
    log_event(f"Found {len(known)} known resources", "INFO")
    
    # Search for new resources
    for query in DISCOVERY_QUERIES:
        new_apis = search_for_new_apis(query)
        for api in new_apis:
            resource = APIResource(
                name=api.get("name", "Unknown"),
                url=api.get("url", ""),
                resource_type=api.get("type", "unknown"),
                free_tier=api.get("free_tier", False),
                limits=api.get("limits", "Unknown"),
                discovered_at=datetime.now().isoformat()
            )
            all_resources.append(resource)
    
    log_event(f"Resource scan complete: {len(all_resources)} total", "INFO")
    return all_resources


def suggest_new_integrations() -> List[str]:
    """
    Analyzes known resources and suggests integrations.
    
    Returns:
        List of suggestion strings
    """
    suggestions = []
    resources = get_known_resources()
    
    # Check which APIs are not currently integrated
    integrated_apis = set()
    
    # Check environment for API keys
    env_patterns = {
        "OPENROUTER": "OpenRouter",
        "GROQ": "Groq",
        "TOGETHER": "Together AI",
        "REPLICATE": "Replicate",
        "HORDE": "AI Horde",
        "HF_": "Hugging Face",
    }
    
    for env_key, api_name in env_patterns.items():
        if any(k for k in os.environ if env_key in k.upper()):
            integrated_apis.add(api_name)
    
    # Find non-integrated resources
    for resource in resources:
        if resource.name not in integrated_apis:
            suggestions.append(
                f"Consider integrating {resource.name} ({resource.url}) - "
                f"Free tier: {resource.limits}"
            )
    
    return suggestions


# Convenience function
def scout() -> Dict[str, Any]:
    """
    Quick resource scouting - returns summary.
    """
    resources = run_resource_scan()
    suggestions = suggest_new_integrations()
    
    return {
        "total_resources": len(resources),
        "known_apis": len(get_known_resources()),
        "suggestions": suggestions,
    }
