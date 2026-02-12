import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)

from core.tool_adapter import get_adapted_tools

async def test_manage_project():
    print("Testing manage_project tool...")
    tools = get_adapted_tools()
    
    if "manage_project" not in tools:
        print("Error: manage_project tool not found.")
        return

    manage_project = tools["manage_project"]
    
    # 1. Test get_plan
    print("\n--- Testing get_plan ---")
    plan = await manage_project(action="get_plan")
    print(f"Plan keys: {plan.keys() if isinstance(plan, dict) else type(plan)}")

    # 2. Test update_status
    print("\n--- Testing update_status ---")
    # We need a task name from the plan
    tasks = plan.get("tasks", [])
    if tasks:
        task_title = tasks[0]["title"]
        print(f"Updating task: {task_title}")
        result = await manage_project(action="update_status", title=task_title, status="active")
        print(f"Result: {result}")
    else:
        print("No tasks found to update.")

    # 3. Test create_checklist (requires Pi Agent)
    print("\n--- Testing create_checklist ---")
    print("This will call Pi Agent. If it fails due to connection, that's expected in some environments.")
    try:
        result = await manage_project(action="create_checklist", goal="Improve social media engagement")
        print(f"Result: {result}")
    except Exception as e:
        print(f"create_checklist failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_manage_project())
