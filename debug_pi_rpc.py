
import asyncio
import logging
import sys
from core.pi_rpc_bridge import PiRPCBridge

# Configure logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger("PiRPCBridge")
logger.setLevel(logging.DEBUG)

async def run_debug():
    print("Initializing Bridge...")
    # Assume we are in project root
    import os
    cwd = os.getcwd()
    print(f"CWD: {cwd}")
    
    bridge = PiRPCBridge(cwd)
    
    async def on_event(data):
        print(f"\n[EVENT] {data}")

    bridge.set_callback(on_event)
    
    print("Starting Bridge...")
    await bridge.start()
    
    if not bridge.running:
        print("Bridge failed to start/stay running.")
        return

    print("Sending prompt...")
    await bridge.send_prompt("Hello, are you there?")
    
    print("Waiting 10 seconds for response...")
    await asyncio.sleep(10)
    
    print("Stopping Bridge...")
    await bridge.stop()

if __name__ == "__main__":
    asyncio.run(run_debug())
