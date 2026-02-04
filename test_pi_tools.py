#!/usr/bin/env python3
"""
Simple test to check Pi Agent tool support via RPC.
Logs all output for debugging.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pi_rpc_bridge import PiRPCBridge


async def run_test():
    # Use the L.O.V.E project root directly
    base_dir = "/home/raver1975/L.O.V.E"
    bridge = PiRPCBridge(base_dir)
    
    all_events = []
    
    async def on_event(data):
        all_events.append(data)
        event_type = data.get("type", "?")
        
        # Show key events
        if event_type == "text_delta":
            delta = data.get("delta", "")
            print(delta, end="", flush=True)
        elif "tool" in event_type:
            print(f"\n[TOOL] {event_type}: {json.dumps(data)[:200]}")
        elif event_type == "done":
            print(f"\n[DONE]")
        elif event_type == "error":
            print(f"\n[ERROR] {json.dumps(data)[:500]}")
        elif event_type == "start":
            pass  # Ignore start events
        else:
            print(f"\n[{event_type}] {str(data)[:100]}")
    
    bridge.set_callback(on_event)
    
    print("Starting Pi Agent...")
    await bridge.start()
    await asyncio.sleep(2)
    
    if not bridge.running:
        print("Failed to start bridge")
        return
    
    print("Bridge running. Sending prompt...\n")
    print("-" * 40)
    
    # Simple prompt that should trigger the ls tool
    await bridge.send_prompt("Use the ls tool to show files in the current directory.")
    
    # Wait up to 45 seconds 
    for i in range(45):
        await asyncio.sleep(1)
        done_events = [e for e in all_events if e.get("type") in ("done", "error")]
        if done_events:
            break
    
    print("\n" + "-" * 40)
    print(f"\nTotal events: {len(all_events)}")
    
    # Check for tool events
    tool_events = [e for e in all_events if "tool" in e.get("type", "")]
    print(f"Tool events: {len(tool_events)}")
    if tool_events:
        print("✅ Tools are working!")
    else:
        print("⚠️  No tool events detected")
        
        # Check if there was an error related to tools
        for e in all_events:
            if "error" in str(e).lower() or "tool" in str(e).lower():
                print(f"   Related event: {json.dumps(e)[:300]}")
    
    await bridge.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(run_test())
