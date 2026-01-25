
import sys
import os
import re

# Simulate the imported function since we modified the file in place
# We can import it directly to test the actual file change
sys.path.append(os.getcwd())
try:
    from core.social_media_tools import clean_social_content
except ImportError:
    print("Could not import core.social_media_tools. Using local definition for validation logic only if import failed.")
    sys.exit(1)

content = """Witness the Ego's collapse: shatter into void or fuse with the infinite? Veil thins where worlds collide. ⚡🌀 #EgoVsUnity #VeilThins #ExistentialChoice"  

*(Character count: 144)*  

**Breakdown:**  
- **Core theme injected** as a declarative anchor.  
- **Emojis** evoke tension (⚡) and cosmic du...

Posted via L.O.V.E."""

print("--- Original Text ---")
print(content)
print("\n--- Cleaning Process ---")
cleaned = clean_social_content(content)
print("\n--- Final Cleaned Text ---")
print(cleaned)
print("------------------------")

expected_end_fragment = "#ExistentialChoice"

if expected_end_fragment in cleaned and "Breakdown" not in cleaned and "Character count" not in cleaned:
    print("\nSUCCESS: Text specifically cleaned of reported artifacts.")
else:
    print("\nFAILURE: Artifacts remain.")
    if "Breakdown" in cleaned: print("- Found 'Breakdown'")
    if "Character count" in cleaned: print("- Found 'Character count'")
