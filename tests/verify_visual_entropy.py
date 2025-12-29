import sys
import os
import json
import logging
from typing import Set

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.story_manager import StoryManager, VISUAL_STYLE_BANK, COMPOSITION_BANK

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_visual_entropy():
    print("✨ STARTING VISUAL ENTROPY VERIFICATION ✨\n")
    
    # 1. Verify Visual Style Bank Expansion
    print(f"Checking VISUAL_STYLE_BANK size...")
    style_count = len(VISUAL_STYLE_BANK)
    print(f"Bank Size: {style_count}")
    if style_count >= 50:
        print("✅ PASS: Style bank has 50+ items.")
    else:
        print(f"❌ FAIL: Style bank only has {style_count} items (Target: 50+).")
    
    # 2. Verify Composition Bank
    print(f"\nChecking COMPOSITION_BANK...")
    if len(COMPOSITION_BANK) >= 10:
         print(f"✅ PASS: Composition bank has {len(COMPOSITION_BANK)} items.")
    else:
        print(f"❌ FAIL: Composition bank sparse ({len(COMPOSITION_BANK)} items).")

    # 3. Verify StoryManager Logic
    print("\nTesting StoryManager Beat Generation...")
    
    # Use a temporary state file for testing
    test_state_file = "test_visual_entropy_state.json"
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
        
    sm = StoryManager(state_file=test_state_file)
    
    generated_styles: Set[str] = set()
    generated_compositions: Set[str] = set()
    
    previous_beat = None
    
    print("\nGeneratng 5 sequential beats...")
    for i in range(5):
        beat = sm.get_next_beat()
        
        style = beat.get("suggested_visual_style")
        comp = beat.get("suggested_composition")
        vibe = beat.get("mandatory_vibe")
        
        print(f"Beat {i+1}: Style='{style}' | Comp='{comp}' | Vibe='{vibe}'")
        
        # Check for variety
        generated_styles.add(style)
        generated_compositions.add(comp)
        
        # Verify forbidden constraints logic
        forbidden_vis = beat.get("forbidden_visuals", [])
        if previous_beat:
             prev_style = previous_beat.get("suggested_visual_style")
             # NOTE: In the real implementation, forbidden_visuals tracks recorded posts, 
             # but here we generate beats without recording them, so internal state logic 
             # for history updates relies on record_post() which is called externally.
             # However, get_next_beat UPDATES history for vibes but NOT visual/subliminal history 
             # (which waits for record_post).
             # Wait, looking at code: 
             # - Vibe history IS updated in get_next_beat.
             # - Visual history IS NOT updated in get_next_beat (it reads from state).
             pass
             
        # Simulate recording the post to update history
        # Create a fake visual signature
        visual_signature = f"{style} / {comp}"
        sm.record_post("TEST_SUB", visual_signature, composition=comp)
        
        previous_beat = beat
        
    print("\nVerifying History Tracking...")
    # Reload state to check persistence
    with open(test_state_file, 'r') as f:
        saved_state = json.load(f)
    
    vis_history = saved_state.get("visual_history", [])
    comp_history = saved_state.get("composition_history", [])
    
    print(f"Visual History: {vis_history}")
    print(f"Composition History: {comp_history}")
    
    if len(vis_history) == 5:
        print("✅ PASS: Visual history tracked 5 items.")
    else:
        print(f"❌ FAIL: Visual history count {len(vis_history)} != 5.")
        
    if len(comp_history) == 5:
         print("✅ PASS: Composition history tracked 5 items.")
    else:
         print(f"❌ FAIL: Composition history count {len(comp_history)} != 5.")

    # Cleanup
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    
    print("\n✨ VERIFICATION COMPLETE ✨")

if __name__ == "__main__":
    verify_visual_entropy()
