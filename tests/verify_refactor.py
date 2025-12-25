import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

print("--- Starting Verification ---")

try:
    print("Importing core.shared_state...")
    import core.shared_state
    print("core.shared_state imported successfully.")
    print(f"Initial memory_manager: {core.shared_state.memory_manager}")

    print("Importing agent...")
    import agent
    print("agent imported successfully.")

    print("Importing core.tools...")
    import core.tools
    print("core.tools imported successfully.")

    print("Importing core.tools_legacy...")
    import core.tools_legacy
    print("core.tools_legacy imported successfully.")

    # Check if love can be imported mostly without checking deps (it might fail due to missing env vars, but imports should work)
    # print("Importing love module (simulating checking for circular deps)...")
    # import love
    # print("love imported.")

    print("--- Verification Successful ---")

except ImportError as e:
    print(f"--- Verification FAILED with ImportError: {e} ---")
except Exception as e:
    print(f"--- Verification FAILED with Exception: {e} ---")
