
import sys
import traceback
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

# Setup attributes
sys.modules['core.llm_api'].run_llm = AsyncMock()

sys.modules['core.text_processing'].intelligent_truncate = AsyncMock()

sys.modules['core.bluesky_api'].post_to_bluesky_with_image = MagicMock()
sys.modules['core.bluesky_api'].get_notifications = MagicMock()
sys.modules['core.bluesky_api'].get_profile = MagicMock()
sys.modules['core.bluesky_api'].get_own_posts = MagicMock()
sys.modules['core.bluesky_api'].get_comments_for_post = MagicMock()
sys.modules['core.bluesky_api'].reply_to_post = MagicMock()
sys.modules['core.bluesky_api'].get_timeline = MagicMock()

sys.modules['core.image_api'].generate_image = AsyncMock()

# Memory Manager
mock_memory = MagicMock()
mock_memory.retrieve_hierarchical_context = MagicMock()
mock_memory.add_episode = AsyncMock()
sys.modules['love'].memory_manager = mock_memory

print("Attempting import...")
try:
    from core.tools_legacy import manage_bluesky
    print("Import SUCCESS")
    
    # Try running it too
    import asyncio
    print("Attempting execution...")
    asyncio.run(manage_bluesky(action='post'))
    print("Execution SUCCESS")

except Exception:
    traceback.print_exc()
