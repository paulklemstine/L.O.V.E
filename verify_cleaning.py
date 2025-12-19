import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from core.social_media_tools import clean_social_content

text = """Behold the threshold where whispers of light are swallowed by silence. 🌌💫 The Ghost emerges—neon pulse in indigo void. A hand reaches, torn between unity and ego's unraveling. 🌀 Choose wisely. #EgoDissolution #UnityOrSilence #DigitalVoid #NeonExistence #HauntingChoices"  

*(278 characters)*"""

cleaned = clean_social_content(text)

print(f"Original: {text!r}")
print(f"Cleaned:  {cleaned!r}")

if "*(278 characters)*" not in cleaned:
    print("SUCCESS: Character count removed.")
else:
    print("FAILURE: Character count NOT removed.")
    sys.exit(1)

# Test normal text
text2 = "Just a normal post. #test"
cleaned2 = clean_social_content(text2)
if cleaned2 == text2:
    print("SUCCESS: Normal text preserved.")
else:
    print(f"FAILURE: Normal text altered: {cleaned2!r}")

