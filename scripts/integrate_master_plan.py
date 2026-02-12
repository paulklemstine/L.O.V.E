import os
import sys
import logging

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)

from core.master_plan_manager import get_master_plan_manager

def main():
    print("Integration Master Plan...")
    manager = get_master_plan_manager()
    if manager.parse_markdown():
        print("Success! Plan parsed and saved to JSON.")
        
        # Verify
        plan = manager.load_plan()
        print(f"Loaded {len(plan.get('epics', []))} epics, {len(plan.get('features', []))} features, {len(plan.get('tasks', []))} tasks.")
    else:
        print("Failed to parse master plan.")

if __name__ == "__main__":
    main()
