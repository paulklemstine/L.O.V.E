
print("Starting...")
import sys
import os
sys.path.append(os.getcwd())
try:
    from core.polly import PollyOptimizer
    print("Imported PollyOptimizer")
except Exception as e:
    print(f"Import failed: {e}")
