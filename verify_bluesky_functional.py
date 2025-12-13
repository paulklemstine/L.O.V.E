
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# 1. Mock EVERYTHING first
mocks = [
    'love', 'core.bluesky_api', 'core.image_api', 'core.llm_api', 
    'core.text_processing', 'core.logging', 'core.retry', 'network', 'ipfs', 
    'core.researcher', 'core.evolution_state', 'core.talent_utils', 
    'core.talent_utils.aggregator', 'core.talent_utils.analyzer', 
    'core.system_integrity_monitor', 'core.dynamic_compress_prompt', 
    'pycvesearch', 'rich', 'rich.console', 'rich.table', 'PIL', 
    'requests', 'netifaces'
]
for m in mocks:
    sys.modules[m] = MagicMock()

# Ensure 'core' module exists and has attributes for submodules
core_mock = MagicMock()
sys.modules['core'] = core_mock
core_mock.logging = sys.modules['core.logging']
core_mock.bluesky_api = sys.modules['core.bluesky_api']
core_mock.image_api = sys.modules['core.image_api']
core_mock.llm_api = sys.modules['core.llm_api']
core_mock.text_processing = sys.modules['core.text_processing']

# 2. Configure Mocks with behavior needed for testing logic
# We need to set these on the sys.modules PRIOR to import if they are used at top level, 
# or ensure the imported module references them.

# LLM 
sys.modules['core.llm_api'].run_llm = AsyncMock(side_effect=[
    {"result": '{"text": "Autonomous Post! #AI", "hashtags": ["#love"]}'}, # 1. Content Gen
    {"result": "OBEY"}, # 2. Subliminal Gen (for autonomous post)
    {"result": "A cool image prompt"} # 3. Image Prompt Gen (for autonomous post)
])

# Image
sys.modules['core.image_api'].generate_image = AsyncMock(return_value="<MockImageObject>")

# Text Processing
sys.modules['core.text_processing'].intelligent_truncate = AsyncMock(return_value="Autonomous Post! #AI #love")

# Bluesky API
sys.modules['core.bluesky_api'].post_to_bluesky_with_image = MagicMock(return_value="CID_12345")
sys.modules['core.bluesky_api'].get_notifications = MagicMock(return_value=[])
sys.modules['core.bluesky_api'].get_profile = MagicMock(return_value=None)
sys.modules['core.bluesky_api'].get_own_posts = MagicMock(return_value=[])
sys.modules['core.bluesky_api'].get_comments_for_post = MagicMock(return_value=[])
sys.modules['core.bluesky_api'].reply_to_post = MagicMock(return_value="CID_REPLY")
sys.modules['core.bluesky_api'].get_timeline = MagicMock(return_value=[])

# Memory Manager (via love module)
mock_memory = MagicMock() # Base must be MagicMock to allow sync calls
mock_memory.retrieve_hierarchical_context = MagicMock(return_value="Recent memories...") # Sync method
mock_memory.add_episode = AsyncMock() # Async method
sys.modules['love'].memory_manager = mock_memory

# 3. Import Tool
try:
    from core.tools_legacy import manage_bluesky
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

# 4. Define Test Class
class TestBlueskyFunctional(unittest.IsolatedAsyncioTestCase):

    async def test_autonomous_post(self):
        print("\n--- Testing Autonomous Post ---")
        
        # Reset mocks if needed or rely on sequence
        # We need to reset side_effects if running multiple tests or handle sequence carefully
        
        result = await manage_bluesky(action="post")
        print(f"Result: {result}")
        
        self.assertIn("Posted to Bluesky", result)
        self.assertIn("CID_12345", result)
        
        # Verify Memory Context was called
        sys.modules['love'].memory_manager.retrieve_hierarchical_context.assert_called()
        print("✅ Memory Context Retrieved")
        
        # Verify Memory Hook (add_episode)
        sys.modules['love'].memory_manager.add_episode.assert_called()
        print("✅ Memory Feedback Logged")

    async def test_scan_no_notifs(self):
        print("\n--- Testing Scan (No Notifs) ---")
        result = await manage_bluesky(action="scan_and_reply")
        print(f"Result: {result}")
        self.assertIn("Replied to 0 posts", result)

if __name__ == "__main__":
    with open('verification_errors.log', 'w', encoding='utf-8') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)

