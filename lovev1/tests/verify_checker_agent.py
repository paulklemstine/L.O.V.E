
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.final_draft_fixer import detect_placeholder_text, validate_image_prompt, DraftIssue

def test_text_placeholder_detection():
    print("--- Testing Text Placeholder Detection ---")
    test_cases = [
        ("This is a clean post.", False),
        ("This post contains MANIPULATIVE_TRIGGER which is bad.", True),
        ("Just a normal YOUR_PHRASE test.", True),
        ("Something [INSERT_HERE] to fix.", True),
    ]
    
    for text, should_detect in test_cases:
        issues = detect_placeholder_text(text)
        detected = len(issues) > 0
        status = "PASS" if detected == should_detect else "FAIL"
        print(f"Text: '{text}' -> Detected: {detected} | Expected: {should_detect} -> {status}")
        if detected:
            for i in issues:
                print(f"  [Issue] {i.details}")

def test_image_prompt_validation():
    print("\n--- Testing Image Prompt Validation ---")
    test_cases = [
        ("A beautiful sunset over the ocean, 8k", False),
        ("A scene with MANIPULATIVE_TRIGGER in neon lights", True),
        ("Placeholder image of a cat", True),
        ("Draft sketch of a robot", True),
    ]
    
    for prompt, should_detect in test_cases:
        issues = validate_image_prompt(prompt)
        detected = len(issues) > 0
        status = "PASS" if detected == should_detect else "FAIL"
        print(f"Prompt: '{prompt}' -> Detected: {detected} | Expected: {should_detect} -> {status}")
        if detected:
            for i in issues:
                print(f"  [Issue] {i.details}")

if __name__ == "__main__":
    test_text_placeholder_detection()
    test_image_prompt_validation()
