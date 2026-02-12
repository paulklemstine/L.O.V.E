import asyncio
import os
import sys
import logging

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from core.pi_rpc_bridge import get_pi_bridge

async def main():
    bridge = get_pi_bridge()
    
    async def on_event(data):
        print(f"EVENT: {data}")

    bridge.set_callback(on_event)
    await bridge.start()
    
    # Wait for init logs
    await asyncio.sleep(5)
    
    print("Sending Hello...")
    await bridge.send_prompt("Hello, just testing.")
    
    await asyncio.sleep(30)
    await bridge.stop()

if __name__ == "__main__":
    asyncio.run(main())
