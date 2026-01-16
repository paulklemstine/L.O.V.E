
import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

print(f"Added to sys.path: {root_dir}")
print("sys.path entries:")
for p in sys.path:
    print(f"  {p}")

try:
    import core
    print(f"Successfully imported core from: {core.__file__}")
    import core.final_draft_fixer
    print(f"Successfully imported core.final_draft_fixer from: {core.final_draft_fixer.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
