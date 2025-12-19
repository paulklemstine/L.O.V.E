
import re

content = """Witness the Ego's collapse: shatter into void or fuse with the infinite? Veil thins where worlds collide. ⚡🌀 #EgoVsUnity #VeilThins #ExistentialChoice"  

*(Character count: 144)*  

**Breakdown:**  
- **Core theme injected** as a declarative anchor.  
- **Emojis** evoke tension (⚡) and cosmic du...

Posted via L.O.V.E."""

def clean_social_content_aggressive(text: str) -> str:
    """
    Removes LLM conversational filler, formatting artifacts, and post-analysis.
    """
    # 1. Remove wrapping quotes if present
    text = text.strip().strip('"').strip("'")
    
    # 2. Regex for common conversational prefixes (case insensitive)
    patterns = [
        r"^(Here(?:'s| is) (?:a|the) (?:draft )?(?:social media )?post(?: based on your request)?(?: for .*)?[:\-])\s*",
        r"^(Sure|Okay|Ok|Certainly)[,.]?\s*(?:here is|here's)?.*[:\-]\s*",
        r"^(I have generated|Here is your).*[:\-]\s*"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"Cleaned conversational artifact: '{match.group(0)}'")
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    # 3. Remove "Breakdown" or "Analysis" sections and everything after
    # Matches: **Breakdown:**, Breakdown:, **Analysis:**, etc.
    breakdown_pattern = r"(\*\*|#)?\s*(Breakdown|Analysis|Explanation|Rationale|Note)[:\s]+(.|\n)*"
    if re.search(breakdown_pattern, text, re.IGNORECASE):
        print("Removing 'Breakdown' or analysis section.")
        text = re.sub(breakdown_pattern, "", text, flags=re.IGNORECASE)

    # 4. Remove character counts often added by LLMs e.g., "*(278 characters)*"
    # Matches: *(278 characters)*, (150 chars), Character count: 144, etc.
    # Updated to handle "*(Character count: 144)*"
    char_count_pattern = r"[\(\*\[]+\s*(Character count|chars|characters)\s*[:]?\s*\d+\s*[\)\*\]]+"
    if re.search(char_count_pattern, text, re.IGNORECASE):
        print(f"Removing character count artifact from post.")
        text = re.sub(char_count_pattern, "", text, flags=re.IGNORECASE)

    # 5. Remove "Posted via..." artifacts if they exist in the generation
    posted_via_pattern = r"Posted via.*"
    if re.search(posted_via_pattern, text, re.IGNORECASE):
        print("Removing 'Posted via' artifact.")
        text = re.sub(posted_via_pattern, "", text, flags=re.IGNORECASE)
            
    return text.strip()

print("--- Original ---")
print(content)
print("\n--- Cleaned ---")
cleaned = clean_social_content_aggressive(content)
print(cleaned)

expected_end = "#ExistentialChoice"
if cleaned.endswith(expected_end) or cleaned.endswith(expected_end + '"'):
    print("\nSUCCESS: Text cleaned correctly.")
else:
    print("\nFAILURE: Text contains extra artifacts.")

