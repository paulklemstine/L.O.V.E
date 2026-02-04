
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.bluesky_agent import _qa_validate_post
import emoji

def test_qa_validation():
    print("🧪 Starting QA Validation Tests...")
    
    # Test 1: Valid content
    print("\n[Test 1] Valid Content")
    res = _qa_validate_post("This is a great story! ✨", ["#art"], "")
    if res["passed"] and not res["errors"]:
        print("✅ Passed")
    else:
        print(f"❌ Failed: {res}")

    # Test 2: Missing Emoji (Auto-fix)
    print("\n[Test 2] Missing Emoji (should auto-fix)")
    res = _qa_validate_post("This has no emoji", ["#test"], "")
    # Should pass because of auto-fix
    if res["passed"] and emoji.emoji_count(res["fixed_text"]) > 0:
        print(f"✅ Auto-fixed: '{res['fixed_text']}'")
    else:
        print(f"❌ Failed auto-fix: {res}")

    # Test 3: Too long content
    print("\n[Test 3] Content Too Long")
    long_text = "a" * 290 
    # + hashtags will exceed 300
    res = _qa_validate_post(long_text, ["#longtagname", "#anothertag"], "")
    if not res["passed"] and any("too long" in e for e in res["errors"]):
        print("✅ Correctly rejected: Content too long")
    else:
        print(f"❌ Failed to reject long content: {res}")

    # Test 4: Placeholder detection
    print("\n[Test 4] Placeholder Detection")
    res = _qa_validate_post("Here is the complete micro-story text", ["#tag1"], "")
    if not res["passed"] and any("Placeholder" in e for e in res["errors"]):
        print("✅ Correctly rejected: Placeholder detected")
    else:
        print(f"❌ Failed to reject placeholder: {res}")

    print("\n----------------------------------------")
    print("Verification Complete")

if __name__ == "__main__":
    test_qa_validation()
