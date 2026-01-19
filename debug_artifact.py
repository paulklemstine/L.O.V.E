import sys
import os
sys.path.append(os.getcwd())
try:
    from core.logging import log_event
    print("Successfully imported log_event")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

large_block = "A" * 600 + "\n" * 15
print(f"Block length: {len(large_block)}, Newlines: {large_block.count(chr(10))}")

try:
    log_event(large_block)
    print("Called log_event")
except Exception as e:
    print(f"log_event raised: {e}")

if os.path.exists("artifacts.log"):
    print("artifacts.log exists!")
    with open("artifacts.log", "r") as f:
        print("Content:", f.read()[:100])
else:
    print("artifacts.log does NOT exist.")
    print("CWD:", os.getcwd())
