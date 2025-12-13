
import sys
from unittest.mock import MagicMock

# Mock everything
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

try:
    from core.tools_legacy import manage_bluesky
    print("SUCCESS: manage_bluesky imported")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
