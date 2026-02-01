
import sys
import os
from pathlib import Path
import json

# Setup path
current_dir = Path(__file__).parent.absolute()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from core.service_manager import ServiceManager

class MockServiceManager(ServiceManager):
    def __init__(self, root_dir, mock_vram):
        super().__init__(root_dir)
        self.mock_vram = mock_vram
        self.STATE_FILE = "state/vw_test_state.json"

    def get_total_vram_mb(self):
        return self.mock_vram

def test_scenario(vram, strikes=0, expected_model_substr=""):
    print(f"\n--- Testing VRAM: {vram}MB, Previous Strikes: {strikes} ---")
    
    # 1. Setup Mock
    mgr = MockServiceManager(root_dir, vram)
    
    # 2. Inject state
    state = {"strikes": strikes}
    mgr._save_state(state)
    
    # 3. Run Selection
    selected = mgr._select_model(vram)
    print(f"Selected: {selected}")
    
    # 4. Verify
    if expected_model_substr in selected:
        print(f"✅ PASS: Selected model contains '{expected_model_substr}'")
    else:
        print(f"❌ FAIL: Expected '{expected_model_substr}' but got '{selected}'")

# Clean up any previous test state
test_state = root_dir / "state/vw_test_state.json"
if test_state.exists():
    os.remove(test_state)

print("🧪 Starting Model Selection Logic Verification (Reasoning Era)")

# Tier 1: Micro
test_scenario(4096, 0, "Qwen2.5-3B-Instruct-AWQ") 

# Tier 2: Small (8GB card)
test_scenario(8192, 0, "Qwen2.5-7B-Instruct-AWQ") 

# Tier 3: Medium (16GB) - Does 8B fit? Yes. Does 30B? No (req >18GB).
# Should pick DeepSeek R1 Distill 14B AWQ.
test_scenario(16000, 0, "DeepSeek-R1-Distill-Qwen-14B-AWQ")

# Tier 4: Large (24GB) - Should fit 32B
test_scenario(24000, 0, "Qwen2.5-32B-Instruct-AWQ")

# Tier 5: Massive (40GB A100) - Should fit DeepSeek R1
test_scenario(40000, 0, "DeepSeek-R1-Distill-Llama-70B")

# Fallback Test: 24GB card crashed on 32B -> Should drop to 14B Reasoning
test_scenario(24000, 1, "DeepSeek-R1-Distill-Qwen-14B-AWQ")

# Clean up
if test_state.exists():
    os.remove(test_state)
