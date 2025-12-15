
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import types

# List of modules to mock
# NOTE: We do NOT mock 'core' itself, as we need to import 'core.tools_legacy' from it.
modules_to_mock = [
    'love',
    # 'core', # Do not mock the top-level package to allow importing real submodules
    'core.bluesky_api',
    'core.image_api',
    'core.llm_api',
    'core.text_processing',
    'core.logging',
    'core.retry',
    'network',
    'ipfs',
    'core.researcher',
    'core.evolution_state',
    'core.talent_utils',
    'core.talent_utils.aggregator',
    'core.talent_utils.analyzer',
    'core.system_integrity_monitor',
    'core.dynamic_compress_prompt',
    'pycvesearch',
    'rich',
    'rich.console',
    'rich.table',
    'rich.color_triplet',
    'PIL',
    'requests',
    'netifaces'
]

# Apply mocks to sys.modules
for m in modules_to_mock:
    sys.modules[m] = MagicMock()

# --- Re-Establish Specific Mocks Required for Tests ---

# 1. core.text_processing
tp = sys.modules['core.text_processing']
tp.intelligent_truncate = AsyncMock(return_value="TRUNCATED")
tp.smart_truncate = AsyncMock(return_value="TRUNCATED")

# 2. core.llm_api
la = sys.modules['core.llm_api']
la.run_llm = AsyncMock()
la.get_llm_api = MagicMock() 

# 3. core.bluesky_api
ba = sys.modules['core.bluesky_api']
ba.post_to_bluesky_with_image = MagicMock()
ba.reply_to_post = MagicMock()
ba.get_timeline = MagicMock()
ba.get_notifications = MagicMock()
ba.get_profile = MagicMock()

# 4. core.image_api
ia = sys.modules['core.image_api']
ia.generate_image = AsyncMock()

# 5. core.logging
sys.modules['core.logging'].log_event = MagicMock()

# Import the tool AFTER mocking
try:
    from core.tools_legacy import manage_bluesky
except ImportError as e:
    print(f"CRITICAL: Failed to import manage_bluesky even after mocking: {e}")
    sys.exit(1)

# --- Original Test Class ---
class TestManageBluesky(unittest.IsolatedAsyncioTestCase):

    async def test_autonomous_post_generation(self):
        print("\n--- Testing Autonomous Post Generation ---")
        
        # Mock dependencies
        mock_memory_manager = AsyncMock()
        mock_memory_manager.retrieve_hierarchical_context.return_value = "Mock Memory Context"
        mock_memory_manager.add_episode = AsyncMock() 
        
        sys.modules['love'].memory_manager = mock_memory_manager
        
        # Mock LLM responses
        mock_run_llm = AsyncMock(side_effect=[
            {"result": '{"text": "Dopamine hit! #test", "hashtags": ["#cool"], "image_prompt": "A cool image"}'}, # Content
            {"result": "WAKE UP"}, # Subliminal
            {"result": "A cool image with WAKE UP in neon"} # Image Prompt
        ])
        sys.modules['core.llm_api'].run_llm = mock_run_llm
        
        # Mock Image Gen
        sys.modules['core.image_api'].generate_image = AsyncMock(return_value="mock_image_object")
        
        # Mock Text Processing
        sys.modules['core.text_processing'].intelligent_truncate = AsyncMock(return_value="Dopamine hit! #test #cool")
        
        # Mock Bluesky API
        sys.modules['core.bluesky_api'].post_to_bluesky_with_image = MagicMock(return_value="SUCCESS_CID")
        
        # RUN
        result = await manage_bluesky(action="post")
        
        print(f"Result: {result}")
        
        # ASSERTIONS
        self.assertIn("Posted to Bluesky", result)
        
        # Verify Memory Context Retrieval
        mock_memory_manager.retrieve_hierarchical_context.assert_called_once()
        print("✅ Memory Context Retrieved")
        
        # Verify LLM called for content
        mock_run_llm.assert_any_call(
            prompt_key="social_media_content_generation", 
            prompt_vars={'type': 'post', 'context': 'Current Memory State: Mock Memory Context'}, 
            purpose="autonomous_post_generation"
        )
        print("✅ LLM Content Gen Called")
        
        # Verify Image Gen
        sys.modules['core.image_api'].generate_image.assert_awaited()
        print("✅ Image Generated")
        
        # Verify Post
        sys.modules['core.bluesky_api'].post_to_bluesky_with_image.assert_called()
        print("✅ Posted to API")
        
        # Verify Memory Feedback Hook
        mock_memory_manager.add_episode.assert_called()
        print("✅ Memory Feedback Hook Triggered")

    async def test_manual_post(self):
        print("\n--- Testing Manual Post w/ Autonomous Image ---")
        
        # Mock dependencies
        mock_memory_manager = AsyncMock()
        mock_memory_manager.add_episode = AsyncMock()
        sys.modules['love'].memory_manager = mock_memory_manager
        
        mock_run_llm = AsyncMock(side_effect=[
            {"result": "OBEY"}, # Subliminal
            {"result": "Image prompt with OBEY"} # Image Prompt Gen
        ])
        sys.modules['core.llm_api'].run_llm = mock_run_llm
        
        sys.modules['core.image_api'].generate_image = AsyncMock(return_value="mock_image_obj")
        sys.modules['core.text_processing'].intelligent_truncate = AsyncMock(return_value="Manual text")
        sys.modules['core.bluesky_api'].post_to_bluesky_with_image = MagicMock(return_value="SUCCESS_CID")

        # RUN
        result = await manage_bluesky(action="post", text="Manual text")
        
        print(f"Result: {result}")
        
        # Assertions
        sys.modules['core.image_api'].generate_image.assert_awaited()
        print("✅ Autonomous Image Gen triggered for Manual Post")
        
        mock_memory_manager.add_episode.assert_called()
        print("✅ Memory Feedback Hook Triggered")

if __name__ == '__main__':
    # Ensure stdout encoding
    if sys.stdout.encoding != 'utf-8':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        
    with open('test_results.log', 'w', encoding='utf-8') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)
