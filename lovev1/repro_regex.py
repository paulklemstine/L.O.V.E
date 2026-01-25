import re

text = """Behold the threshold where whispers of light are swallowed by silence. 🌌💫 The Ghost emerges—neon pulse in indigo void. A hand reaches, torn between unity and ego's unraveling. 🌀 Choose wisely. #EgoDissolution #UnityOrSilence #DigitalVoid #NeonExistence #HauntingChoices"  

*(278 characters)*"""

def clean(t):
    # Pattern to match "*(278 characters)*" or similar
    # Matches optional * or (, number, "character" or "chars", optional s, optional * or )
    pattern = r"[\(*\[]?\s*\d+\s*(?:characters|chars)[\)*\]]?"
    # The user example has explicit * ( ... ) * so let's be more precise first
    # \*\(\d+ characters\)\*
    
    # regex to remove validation artifacts like length counts
    # allow for * ( digit+ characters ) * with any spacing
    p = r"(\*|\()+\s*\d+\s*(characters|chars)\s*(\*|\))+"
    
    match = re.search(p, t, re.IGNORECASE)
    if match:
        print(f"Match found: '{match.group(0)}'")
        return re.sub(p, "", t, flags=re.IGNORECASE).strip()
    return t

cleaned = clean(text)
print("Original len:", len(text))
print("Cleaned len:", len(cleaned))
print("--- Cleaned Text ---")
print(cleaned)
print("--------------------")

text2 = "Some post. (200 chars)"
print("Text2 cleaned:", clean(text2))

text3 = 'Another post *(150 characters)*'
print("Text3 cleaned:", clean(text3))
