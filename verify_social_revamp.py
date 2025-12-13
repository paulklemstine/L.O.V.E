
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# aggressively mock everything to prevent ImportErrors
mocks = [
    'love',
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
    'PIL',
    'requests',
    'netifaces'
]

for m in mocks:
    sys.modules[m] = MagicMock()

# Now import the tool to test
# Setup attributes explicitly to avoid AttributeError
import types
# core.text_processing
tp = types.ModuleType('core.text_processing')
tp.intelligent_truncate = AsyncMock(return_value="TRUNCATED")
tp.smart_truncate = AsyncMock(return_value="TRUNCATED")
sys.modules['core.text_processing'] = tp

# core.llm_api
la = types.ModuleType('core.llm_api')
la.run_llm = AsyncMock()
sys.modules['core.llm_api'] = la

# core.bluesky_api
ba = types.ModuleType('core.bluesky_api')
ba.post_to_bluesky_with_image = MagicMock()
ba.reply_to_post = MagicMock()
ba.get_timeline = MagicMock()
ba.get_notifications = MagicMock()
ba.get_profile = MagicMock()
sys.modules['core.bluesky_api'] = ba

# core.image_api
ia = types.ModuleType('core.image_api')
ia.generate_image = AsyncMock()
sys.modules['core.image_api'] = ia

from core.tools_legacy import manage_bluesky

class TestManageBluesky(unittest.IsolatedAsyncioTestCase):

    async def test_autonomous_post_generation(self):
        print("\n--- Testing Autonomous Post Generation ---")
        
        # Mock dependencies
        mock_memory_manager = AsyncMock()
        mock_memory_manager.retrieve_hierarchical_context.return_value = "Mock Memory Context"
        mock_memory_manager.add_episode = AsyncMock() # Mock add_episode
        
        sys.modules['love'].memory_manager = mock_memory_manager
        
        # Mock LLM responses
        # 1. Content Generation
        # 2. Subliminal Gen
        # 3. Image Prompt Gen
        
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
        # Should NOT retrieve memory context for content gen, but should for image gen?
        # In current logic, if text is provided, it skips content gen.
        
        # Should generate image? Yes, autonomous image gen is triggered if logic flows there.
        # Wait, the tool logic:
        # if not text: -> generate text
        # ... logic continues ...
        # if image_path: ... else: -> generate image (if prompted or always? Current logic: if autonomous (no text provided?) OR if we just want to?)
        # Let's check the code:
        # "if not text:" ... generates text ...
        # ...
        # "else: # Generate Image if requested OR if autonomous"
        # Wait, if text IS provided, does it skip image gen unless image_prompt is provided?
        # My refactored code checks:
        # "if image_path: ... else: ... # We catch ANY case where we might generate an image"
        # It proceeds to Subliminal Gen Step 1.
        # Then Step 2: "if final_img_prompt:"
        # Wait, where does `final_img_prompt` come from if `image_prompt` was None and we provided text?
        # "else: # Generate from scratch ... img_gen_prompt = ... final_img_prompt = img_res..."
        # So YES, it autonomously generates an image even if text is provided manually.
        
        sys.modules['core.image_api'].generate_image.assert_awaited()
        print("✅ Autonomous Image Gen triggered for Manual Post")
        
        mock_memory_manager.add_episode.assert_called()
        print("✅ Memory Feedback Hook Triggered")

if __name__ == '__main__':
    with open('test_results.log', 'w', encoding='utf-8') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)

