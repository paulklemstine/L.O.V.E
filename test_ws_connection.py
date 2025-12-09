import asyncio
import websockets
import sys

async def test():
    uri = "ws://127.0.0.1:8082"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            await websocket.send(b"\x00") # Dummy package (will likely cause error or ignore)
            print("Sent data.")
            await asyncio.sleep(0.5)
            print("Closing.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
