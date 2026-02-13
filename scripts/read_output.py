
import sys
import os

filename = "test_output.txt"
if not os.path.exists(filename):
    print(f"File {filename} not found.")
    sys.exit(1)

try:
    # Try reading as utf-16 (common for PowerShell redirects)
    with open(filename, 'r', encoding='utf-16') as f:
        print(f.read())
except UnicodeError:
    try:
        # Fallback to utf-8 if not utf-16
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading file: {e}")
