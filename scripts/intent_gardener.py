import asyncio
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock imports if core dependencies are missing in this context, 
# but in real usage we expect them.
try:
    from core.intent_layer.loader import IntentLoader
    from core.llm_api import run_llm
    from core.logging import log_event
except ImportError:
    print("Failed to import core modules. Ensure setup is correct.")
    sys.exit(1)

async def check_diff_against_intent(diff_text: str, file_path: str):
    """
    Checks if a diff violates the Intent Layer context for the given file.
    """
    if not diff_text.strip():
        return

    # Get context stack
    stack = IntentLoader.get_context_stack(os.path.abspath(file_path))
    context_str = IntentLoader.format_context_stack(stack)
    
    if not context_str:
        return

    prompt = f"""
    You are an Intent Gardener. Your job is to review code changes and ensure they adhere to the project's Intent Layer (AGENTS.md).

    *** CONTEXT (AGENTS.md) ***
    {context_str}

    *** CODE CHANGES (Diff) ***
    File: {file_path}
    {diff_text}

    *** INSTRUCTIONS ***
    Analyze the diff against the context.
    1. Does this change violate any "Invariant" or "Anti-pattern"?
    2. Does this change introduce a new pattern that SHOULD be documented in AGENTS.md?
    
    If YES to either, return a concise report.
    If NO violation and no update needed, return "PASS".
    """

    try:
        result = await run_llm(prompt, purpose="intent_gardener")
        response = result.get('result', '').strip()
        
        if "PASS" not in response and len(response) > 10:
            print(f"\n[GARDENER REPORT] File: {file_path}")
            print(response)
            print("-" * 40)
            
    except Exception as e:
        print(f"Error running LLM for {file_path}: {e}")

async def main():
    # 1. Get changed files
    try:
        # Get diff against main or HEAD~1
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        output = subprocess.check_output(cmd, text=True).strip()
        files = output.split('\n')
    except Exception as e:
        print(f"Failed to get git diff: {e}")
        return

    tasks = []
    
    for relative_path in files:
        if not relative_path.endswith('.py'): # Focus on Python for now
            continue
            
        full_path = PROJECT_ROOT / relative_path
        if not full_path.exists():
            continue
            
        # Get the actual diff for this file
        try:
            diff_cmd = ["git", "diff", "HEAD~1", "HEAD", "--", relative_path]
            diff_text = subprocess.check_output(diff_cmd, text=True)
            
            tasks.append(check_diff_against_intent(diff_text, str(full_path)))
        except Exception:
            continue
            
    if tasks:
        print(f"Running Intent Gardener on {len(tasks)} files...")
        await asyncio.gather(*tasks)
    else:
        print("No Python changes detected.")

if __name__ == "__main__":
    asyncio.run(main())
