
import sys
import os
import json
from langchain_core.messages import AIMessage, HumanMessage

# Add project root to path
sys.path.append(os.getcwd())

# Mock core.logging
class MockLogging:
    def log_event(self, msg, level):
        print(f"[{level}] {msg}")

import core.nodes.reasoning
core.nodes.reasoning.core.logging = MockLogging()

from core.nodes.reasoning import _parse_reasoning_response

def test_zombie_logic():
    print("--- Testing Zombie Response Logic ---")
    
    # Simulate a "Zombie" response: Valid JSON, has thought, missing action/final
    zombie_response = json.dumps({
        "thought": "I need to check the file system.",
        "action": None, 
        "final_response": None
    })
    
    print(f"Input Response: {zombie_response}")
    
    # 1. Parse
    thought, parsed_tool_calls, final_response = _parse_reasoning_response(zombie_response)
    print(f"Parsed: thought='{thought}', tool_calls={parsed_tool_calls}, final='{final_response}'")
    
    # 2. Simulate reason_node logic
    stop_reason = None
    messages_to_return = []
    
    if parsed_tool_calls:
        stop_reason = "tool_call"
    
    # Logic extracted from reason_node
    if not parsed_tool_calls and not final_response and thought:
         print(">>> DETECTION TRIGGERED <<<")
         stop_reason = "retry_format_error"
         correction_msg = HumanMessage(content="SYSTEM: You provided a 'thought' but NO 'action'...")
         messages_to_return.append(correction_msg)
    else:
        print(">>> DETECTION FAILED <<<")

    # 3. Assertions
    if stop_reason == "retry_format_error":
        print("SUCCESS: Logic correctly identified zombie response and set stop_reason='retry_format_error'")
    else:
        print(f"FAILURE: Logic failed. stop_reason={stop_reason}")

if __name__ == "__main__":
    test_zombie_logic()
