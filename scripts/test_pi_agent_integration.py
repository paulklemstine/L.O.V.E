
import asyncio
import os
import sys
import json
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pi_rpc_bridge import get_pi_bridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

async def test_agent_integration():
    print("🚀 Starting Pi Agent Integration Test...")
    
    bridge = get_pi_bridge()
    test_filename = "pi_agent_integration_test.txt"
    test_content = "Integration Test Successful"
    
    # Ensure clean state
    if os.path.exists(test_filename):
        try:
            os.remove(test_filename)
        except OSError:
            pass

        
    try:
        # Start the bridge
        print("1. Starting Bridge...")
        await bridge.start()
        
        # Wait for ready
        print("   Waiting for agent ready...")
        # We can implement a waiter or just sleep
        await asyncio.sleep(5) 
        
        # prompt
        prompt = f"""
You are L.O.V.E.
Please perform the following action immediately:
Create a file named '{test_filename}' with the content '{test_content}'.
Use the 'write' tool.
Do not ask for permission. Just do it.
"""
        print(f"2. Sending Prompt: {prompt.strip()[:50]}...")
        
        # Send prompt
        await bridge.send_prompt(prompt)
        
        # Wait for response/action
        print("3. Waiting for 30 seconds for agent to act...")
        await asyncio.sleep(30)
        
        # Check result
        print("4. Checking for file existence...")
        if os.path.exists(test_filename):
            print(f"✅ SUCCESS: File '{test_filename}' was created!")
            with open(test_filename, 'r') as f:
                content = f.read()
            print(f"   Content: {content}")
        else:
            print(f"❌ FAILURE: File '{test_filename}' was NOT created.")
            print("   This indicates the agent did not execute the tool, or execution failed.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        print("5. Stopping Bridge...")
        await bridge.stop()
        
        # Cleanup - kept for manual inspection
        # if os.path.exists(test_filename):
        #     os.remove(test_filename)

if __name__ == "__main__":
    asyncio.run(test_agent_integration())
