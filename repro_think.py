import re

def strip_thinking_tags_old(response: str) -> str:
    if not response:
        return ""
    # Current logic in llm_parser.py
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    return response.strip()

def strip_thinking_tags_new(response: str) -> str:
    if not response:
        return ""
    # Improved logic: handle unclosed tags at the end
    # Using [\s\S] instead of . with DOTALL to be explicit, but DOTALL is fine too.
    # The key is (?:</think>|$)
    response = re.sub(r'<think>.*?(?:</think>|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    return response.strip()

test_cases = [
    ("<think>Thought process</think> Result", "Result"),
    ("<think>Unclosed thought process", ""), # Should be empty in new, fails in old
    ("Result only", "Result only"),
    ("<think>Thought</think> Result <think>More thought</think>", "Result"),
    ("<think>Truncated thought...", ""),
    ("Some content <think>thought</think>", "Some content")
]

print("Testing Regex Logic:")
print("-" * 40)

for inp, expected in test_cases:
    got_old = strip_thinking_tags_old(inp)
    got_new = strip_thinking_tags_new(inp)
    
    print(f"Input:    '{inp}'")
    print(f"Old Res:  '{got_old}'")
    print(f"New Res:  '{got_new}'")
    
    if got_new == expected:
        print("Status:   PASS (New matches expectation)")
    elif got_new == "" and expected == "":
         print("Status:   PASS (New matches expectation)")
    else:
        print(f"Status:   FAIL (Expected '{expected}')")
        
    print("-" * 20)
