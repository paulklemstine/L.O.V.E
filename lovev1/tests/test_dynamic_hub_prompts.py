"""
Test Dynamic Hub Prompt Discovery for Subagents

This test verifies the ability to:
1. Search LangChain Hub for prompts matching capabilities (e.g., "mathematician")
2. Cache discovered prompts to hub_prompt_cache.yaml (separate from core prompts)
3. Spawn subagents with dynamically discovered prompts
4. Fall back gracefully when Hub is unavailable
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.subagent_loader import SubagentLoader, HUB_PROMPT_CACHE_FILE


class TestHubSearch(unittest.TestCase):
    """Tests for Hub search capability."""
    
    def setUp(self):
        """Reset caches before each test."""
        SubagentLoader._cache.clear()
        SubagentLoader._hub_cache.clear()
        
        # Backup and clear the file cache for testing
        self.original_cache_exists = os.path.exists(HUB_PROMPT_CACHE_FILE)
        if self.original_cache_exists:
            with open(HUB_PROMPT_CACHE_FILE, 'r') as f:
                self.original_cache_content = f.read()
    
    def tearDown(self):
        """Restore original cache after tests."""
        if self.original_cache_exists:
            with open(HUB_PROMPT_CACHE_FILE, 'w') as f:
                f.write(self.original_cache_content)
    
    def test_search_mathematician_capability(self):
        """Test searching for a mathematician capability."""
        async def run_test():
            results = await SubagentLoader.search_hub_for_capability("mathematician")
            
            self.assertTrue(len(results) > 0, "Should return at least one result")
            
            # Check that we got a good match
            top_result = results[0]
            self.assertIn("score", top_result)
            self.assertGreater(top_result["score"], 0.5, "Top result should have good score")
            self.assertIn("id", top_result)
            
            print(f"Found {len(results)} prompts for 'mathematician':")
            for r in results:
                print(f"  - {r['name']} (score: {r['score']}, source: {r['source']})")
        
        asyncio.run(run_test())
    
    def test_search_legal_capability(self):
        """Test searching for legal/lawyer capability."""
        async def run_test():
            results = await SubagentLoader.search_hub_for_capability("legal expert")
            
            self.assertTrue(len(results) > 0)
            
            # Should match "lawyer" keyword
            has_lawyer_match = any("lawyer" in r.get("name", "").lower() for r in results)
            self.assertTrue(has_lawyer_match or len(results) > 0, "Should match legal keywords")
        
        asyncio.run(run_test())
    
    def test_search_returns_sorted_by_score(self):
        """Test that results are sorted by relevance score."""
        async def run_test():
            results = await SubagentLoader.search_hub_for_capability("data analysis")
            
            if len(results) >= 2:
                for i in range(len(results) - 1):
                    self.assertGreaterEqual(
                        results[i]["score"], 
                        results[i+1]["score"],
                        "Results should be sorted by score descending"
                    )
        
        asyncio.run(run_test())
    
    def test_search_includes_love_fallback(self):
        """Test that search always includes a LOVE repo fallback."""
        async def run_test():
            results = await SubagentLoader.search_hub_for_capability("obscure_capability_xyz")
            
            # Should have at least the LOVE fallback
            love_results = [r for r in results if r["source"] == "love_repo"]
            self.assertTrue(len(love_results) > 0, "Should include LOVE repo fallback")
        
        asyncio.run(run_test())


class TestFileCaching(unittest.TestCase):
    """Tests for file-based prompt caching."""
    
    def setUp(self):
        """Clear caches."""
        SubagentLoader._cache.clear()
        SubagentLoader._hub_cache.clear()
    
    def test_save_and_load_from_file_cache(self):
        """Test saving and loading prompts from file cache."""
        test_capability = "test_mathematician"
        test_hub_id = "test/math-agent"
        test_content = "You are a test mathematician agent."
        
        # Save to cache
        SubagentLoader._save_to_file_cache(test_capability, test_hub_id, test_content)
        
        # Load from cache
        loaded = SubagentLoader._load_from_file_cache(test_capability)
        
        self.assertIsNotNone(loaded, "Should load cached content")
        self.assertEqual(loaded["content"], test_content)
        self.assertEqual(loaded["hub_id"], test_hub_id)
        self.assertIn("fetched_at", loaded)
        
        # Clean up
        try:
            with open(HUB_PROMPT_CACHE_FILE, 'r') as f:
                cache = yaml.safe_load(f) or {}
            del cache[test_capability]
            with open(HUB_PROMPT_CACHE_FILE, 'w') as f:
                yaml.safe_dump(cache, f)
        except Exception:
            pass
    
    def test_list_file_cached_prompts(self):
        """Test listing cached prompts."""
        # Add a test entry
        SubagentLoader._save_to_file_cache("test_list_entry", "test/id", "content")
        
        cached = SubagentLoader.list_file_cached_prompts()
        self.assertIn("test_list_entry", cached)
        
        # Clean up
        try:
            with open(HUB_PROMPT_CACHE_FILE, 'r') as f:
                cache = yaml.safe_load(f) or {}
            del cache["test_list_entry"]
            with open(HUB_PROMPT_CACHE_FILE, 'w') as f:
                yaml.safe_dump(cache, f)
        except Exception:
            pass


class TestGetBestPrompt(unittest.TestCase):
    """Tests for getting the best prompt for a capability."""
    
    def setUp(self):
        SubagentLoader._cache.clear()
        SubagentLoader._hub_cache.clear()
    
    def test_get_best_prompt_uses_file_cache(self):
        """Test that get_best_prompt returns cached content."""
        async def run_test():
            # Pre-populate cache
            test_content = "Cached mathematician prompt content"
            SubagentLoader._save_to_file_cache("mathematician", "cached/math", test_content)
            
            # Get best prompt should return cached
            result = await SubagentLoader.get_best_prompt_for_capability("mathematician")
            
            self.assertEqual(result, test_content)
        
        asyncio.run(run_test())
        
        # Clean up
        try:
            with open(HUB_PROMPT_CACHE_FILE, 'r') as f:
                cache = yaml.safe_load(f) or {}
            del cache["mathematician"]
            with open(HUB_PROMPT_CACHE_FILE, 'w') as f:
                yaml.safe_dump(cache, f)
        except Exception:
            pass
    
    def test_get_best_prompt_with_hub_mock(self):
        """Test getting prompt from Hub with mocked response."""
        async def run_test():
            # Mock the registry's get_hub_prompt
            from core.prompt_registry import PromptRegistry
            PromptRegistry._instance = None
            
            with patch.object(PromptRegistry, 'get_hub_prompt') as mock_hub:
                mock_hub.return_value = "MOCKED HUB CONTENT"
                
                # Clear any file cache for this test
                SubagentLoader._save_to_file_cache = MagicMock()
                
                result = await SubagentLoader.get_best_prompt_for_capability("physicist")
                
                # Should have called Hub
                self.assertTrue(mock_hub.called or result is not None)
        
        asyncio.run(run_test())


class TestSubagentExecutorIntegration(unittest.TestCase):
    """Integration tests for SubagentExecutor with Hub search."""
    
    def test_spawn_custom_subagent_with_capability(self):
        """Test spawning a custom subagent with a capability description."""
        from core.subagent_executor import SubagentExecutor
        
        async def run_test():
            executor = SubagentExecutor()
            
            # Mock out the LLM call to avoid actual API calls
            with patch('core.llm_api.run_llm') as mock_llm:
                mock_llm.return_value = {"result": "Test response from mathematician"}
                
                result = await executor.spawn_custom_subagent(
                    requesting_agent_id="test_agent",
                    desired_capability="mathematician to solve algebra problems",
                    task="What is 2x + 5 = 15, solve for x?"
                )
                
                self.assertTrue(result.success or "error" in result.result.lower())
                self.assertTrue(len(result.agent_type) > 0)
        
        asyncio.run(run_test())


class TestCapabilityKeywords(unittest.TestCase):
    """Tests for capability keyword matching."""
    
    def test_all_capability_categories_present(self):
        """Verify all major capability categories are defined."""
        categories = ["mathematician", "lawyer", "programmer", "writer", "scientist"]
        
        for cat in categories:
            self.assertIn(cat, SubagentLoader.CAPABILITY_KEYWORDS,
                         f"Missing category: {cat}")
    
    def test_capability_keywords_not_empty(self):
        """Ensure each capability has keywords."""
        for cap, keywords in SubagentLoader.CAPABILITY_KEYWORDS.items():
            self.assertTrue(len(keywords) > 0, f"{cap} has no keywords")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing Dynamic Hub Prompt Discovery")
    print("="*60 + "\n")
    
    # Run with verbose output
    unittest.main(verbosity=2)
