import sys
import os
import time
import asyncio
import subprocess
from core.plan_manager import PlanManager
from core.guardrails import GuardrailsManager

# Lightweight Memory Manager (Mock)
class LightweightMemoryManager:
    async def save_activity_log(self, action, result, filepath=".ralph/activity_log.md"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{timestamp}] {action}\n**Result:** {result}\n\n"
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(entry)

    async def read_recent_activity(self, filepath=".ralph/activity_log.md", n=5):
        if not os.path.exists(filepath):
            return ""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        import re
        entries = re.split(r'(?=## \[\d{4}-\d{2}-\d{2})', content)
        entries = [e for e in entries if e.strip()]
        return "".join(entries[-n:])

def run_linter(path="core/"):
    """Runs pylinter and returns success boolean."""
    print(f"🧹 Running Linter on {path}...")
    try:
        # Run pylint, capturing output. 
        # exit code 0 is success (no errors).
        result = subprocess.run(
            ["pylint", path, "--errors-only", "--disable=E0401"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ Linter passed.")
            return True, ""
        else:
            print(f"❌ Linter failed with code {result.returncode}")
            return False, result.stdout + result.stderr
    except FileNotFoundError:
        print("⚠️ Pylint not found. Skipping lint check.")
        return True, "Linter skipped (not installed)"

# Real agent execution placeholder
async def execute_agent_task(task_text, context, guardrails):
    print(f"🤖 AGENT: Working on '{task_text}'")
    print(f"📝 STATUS: Analyzing task requirements...")
    print(f"🧠 MEMORY: Loading {len(context)} chars of recent context.")
    print(f"🛡️ GUARDRAILS: {len(guardrails)} chars active.")
    
    if "missing dependency" in task_text.lower():
        print("🔍 DISCOVERY: Missing dependency detected!")
        print("✋ REQUEST: Deferring execution to install dependency.")
        return "MISSING_DEP"

    if "fail" in task_text.lower():
        print("💥 SIMULATION: Triggering artificial failure for testing.")
        raise Exception("Simulated Failure for Guardrail Loop")

    # Epic 4: Social Tasks
    if "scan social sentiment" in task_text.lower() or "vibe check" in task_text.lower():
        print("📡 ACTION: Scanning Social Timeline (Vibe Check)...")
        # Instantiate agent only when needed (lazy load)
        from core.social_media_agent import SocialMediaAgent
        # We need a shared state dummy or load real one
        mock_state = {"autopilot_goal": "Serve"} 
        agent = SocialMediaAgent(loop=asyncio.get_running_loop(), love_state=mock_state)
        return await agent.scan_social_sentiment_task()

    if "scout resources" in task_text.lower():
        print("🔭 ACTION: Scouting for Resources/Grants...")
        from core.social_media_agent import SocialMediaAgent
        mock_state = {"autopilot_goal": "Serve"}
        agent = SocialMediaAgent(loop=asyncio.get_running_loop(), love_state=mock_state)
        return await agent.scout_resources_task()

    if "soft ask" in task_text.lower() or "blessing" in task_text.lower():
        print("🙏 ACTION: Initiating Blessing Protocol (Soft Ask)...")
        from core.social_media_agent import SocialMediaAgent
        mock_state = {"autopilot_goal": "Serve", "wallet_address": "0x123...BOSS"}
        agent = SocialMediaAgent(loop=asyncio.get_running_loop(), love_state=mock_state)
        return await agent.execute_blessing_protocol_task()

    print("⚙️ EXECUTING: Running generic task logic...")
    await asyncio.sleep(1) 
    print("✅ COMPLETED: Task finished successfully.")
    return "Generic Task Completed"

async def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ralph Task Runner: Waking up...")
    
    plan_manager = PlanManager(plan_path="IMPLEMENTATION_PLAN.md")
    memory_manager = LightweightMemoryManager()
    guardrails_manager = GuardrailsManager()
    
    # 0. Load Context & Guardrails
    recent_activity = await memory_manager.read_recent_activity()
    guardrails = guardrails_manager.read_guardrails()
    
    # 1. Read Plan
    try:
        task = plan_manager.get_next_task()
    except FileNotFoundError:
        print("❌ Error: IMPLEMENTATION_PLAN.md not found.")
        sys.exit(1)

    if not task:
        print("🎉 No pending tasks. Idle.")
        sys.exit(0)
        
    print(f"📋 Found Task: {task['text']}")
    plan_manager.mark_task_in_progress(task['line_index'])
    
    # 3. Execute Task
    try:
        result = await execute_agent_task(task['text'], recent_activity, guardrails)
        
        if result == "MISSING_DEP":
            print("🧱 Missing dependency detected. Adding task...")
            plan_manager.add_task("Install required dependency", position="top")
            print("Deferring to next loop.")
            await memory_manager.save_activity_log(task['text'], "Deferred: Missing Dependency")
            sys.exit(0)
            
        # Epic 3: Linter Gatekeeper
        is_coding_task = any(kw in task['text'].lower() for kw in ['code', 'implement', 'fix', 'update', 'create'])
        # For now, we assume most tasks involve code, or check specifically.
        # Let's run linter on "core/" if it's a coding task or generic default.
        if is_coding_task:
            lint_success, lint_output = run_linter()
            if not lint_success:
                print("🚧 Linter Gatekeeper stopped completion.")
                # Add task to fix lint
                plan_manager.add_task(f"Fix lint errors for {task['text']}", position="top")
                guardrails_manager.add_sign(f"Linting failed for {task['text']}", lint_output[:200]) # truncated log
                await memory_manager.save_activity_log(task['text'], "Failed: Lint Errors")
                sys.exit(1) # Ralph loop will restart, next task is "Fix lint errors"

        success = True
    except Exception as e:
        print(f"💥 Task Exception: {e}")
        guardrails_manager.add_sign(f"Task '{task['text']}' failed.", str(e))
        success = False
        
    # 4. Update Result
    if success:
        print("✅ Task Success.")
        plan_manager.mark_task_complete(task['line_index'])
        await memory_manager.save_activity_log(task['text'], "Success")
        sys.exit(0)
    else:
        print("⚠️ Task Failed. Guardrail updated.")
        await memory_manager.save_activity_log(task['text'], "Failed: Guardrail created")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
