
import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock env vars for test if not present
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = "ls__test_key" # Dummy key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "love-agent-test"

from core.tracing import init_tracing, log_feedback
from core.user_story_validator import UserStoryValidator

async def test_tracing():
    print("Initializing tracing...")
    init_tracing()
    
    print("Testing feedback logging...")
    try:
        # Just check it doesn't crash on invalid run_id
        log_feedback("test_run_id", "test_key", 1.0, "Test comment")
        print("Feedback logging call executed (success/failure details hidden by try-except in logging func, but no crash).")
    except Exception as e:
        print(f"Feedback logging crashed: {e}")
        return

    print("Testing UserStoryValidator instrumentation...")
    validator = UserStoryValidator()
    # Valid story
    story = """# Test Story
**As a** tester
**I want** to verify tracing
**So that** I know it works
## Acceptance Criteria
- [ ] tracing works
- [ ] feedback works
- [ ] validation works
## Technical Specification
Files to modify: none
Implementation: none
"""
    validation = validator.validate(story)
    print(f"Validation result: {validation.is_valid}")
    # We can't easily verify the trace was sent without checking network or mocking client, 
    # but successful execution means the import and call didn't crash.

    print("Verification script finished successfully.")

if __name__ == "__main__":
    asyncio.run(test_tracing())
