import os
import re
import sys
from pathlib import Path

# Constants
INTENT_FILENAME = "AGENTS.md"
REQUIRED_SECTIONS = ["Purpose", "Anti-patterns"]
PYTHON_FILE_THRESHOLD = 10

def lint_intent_layer(root_dir: str):
    """
    Simulates a linting process for the Intent Layer.
    """
    root_path = Path(root_dir).resolve()
    print(f"Linting Intent Layer starting at: {root_path}")
    
    issues_found = 0
    
    for current_path, dirs, files in os.walk(root_path):
        current_path_obj = Path(current_path)
        
        # Skip hidden directories and internal envs
        if any(part.startswith('.') for part in current_path_obj.parts):
            continue
        if "venv" in str(current_path_obj) or "__pycache__" in str(current_path_obj):
            continue

        # Check 1: Missing AGENTS.md in dense directories
        py_files = [f for f in files if f.endswith('.py')]
        intent_file = current_path_obj / INTENT_FILENAME
        
        if len(py_files) > PYTHON_FILE_THRESHOLD and not intent_file.exists():
            print(f"[WARNING] Directory '{current_path_obj.relative_to(root_path)}' has {len(py_files)} Python files but no {INTENT_FILENAME}.")
            issues_found += 1
            
        # Check 2: AGENTS.md content validation
        if intent_file.exists():
            try:
                content = intent_file.read_text(encoding='utf-8')
                
                # Header check
                for section in REQUIRED_SECTIONS:
                    # Fix: Use simpler regex construction to avoid f-string escaping hell
                    # We want to match: Start of line, one or more #, whitespace, section name
                    pattern = r"^#+\s+" + re.escape(section)
                    if not re.search(pattern, content, re.MULTILINE | re.IGNORECASE) and not re.search(f"\\*\\*{section}", content, re.IGNORECASE):
                         # Simple check, could be more robust
                         # Allowing **Purpose** or # Purpose
                         if section.lower() not in content.lower():
                             print(f"[ERROR] {intent_file.relative_to(root_path)} missing required section: '{section}'")
                             issues_found += 1
                
                # Link check
                links = re.findall(r'\[.*?\]\((.*?)\)', content)
                for link in links:
                    if link.startswith('http') or link.startswith('#'):
                        continue
                        
                    # Resolve relative link
                    link_path = (current_path_obj / link).resolve()
                    if not link_path.exists():
                        print(f"[ERROR] {intent_file.relative_to(root_path)} has broken link: '{link}' -> '{link_path}'")
                        issues_found += 1
                        
            except Exception as e:
                print(f"[ERROR] Failed to read {intent_file}: {e}")
                issues_found += 1

    if issues_found:
        print(f"\nLinting complete. Found {issues_found} issues.")
        sys.exit(1)
    else:
        print("\nLinting complete. No issues found. The Intent Layer is healthy.")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        # Default to project root (assuming script is in /scripts)
        root = Path(__file__).resolve().parent.parent
    
    lint_intent_layer(root)
