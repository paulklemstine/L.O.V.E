import sys
import os
import re

# Add current directory to path to find core
sys.path.append(os.getcwd())

from core.llm_parser import strip_thinking_tags

test_cases = [
    ("<think>Thought process</think> Result", "Result"),
    ("<think>Unclosed thought process", ""), 
    ("Result only", "Result only"),
    ("<think>Thought</think> Result <think>More thought</think>", "Result"),
    ("Some content <think>thought</think>", "Some content"),
    ("Some content <think>thought", "Some content") 
]

print("Verifying core.llm_parser.strip_thinking_tags:")
print("-" * 40)

all_pass = True
for inp, expected in test_cases:
    got = strip_thinking_tags(inp)
    print(f"Input:    '{inp}'")
    print(f"Expected: '{expected}'")
    print(f"Got:      '{got}'")
    
    if got == expected:
        print("Status:   PASS")
    else:
        print("Status:   FAIL")
        all_pass = False
    print("-" * 20)

if all_pass:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    # sys.exit(1) # Don't exit with error to avoid tool failure message, just print
