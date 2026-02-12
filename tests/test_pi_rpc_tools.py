#!/usr/bin/env python3
"""
Test script to verify Pi Agent RPC bridge and tool usage.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pi_rpc_bridge import PiRPCBridge, get_pi_bridge
from core.logger import setup_logging

async def test_rpc_with_tools():
    """Test RPC connection and tool usage."""
    # setup_logging(verbose=False)
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bridge = PiRPCBridge(base_dir)
    
    events = []
    errors = []
    
    async def on_event(data):
        """Callback for RPC events."""
        event_type = data.get("type", "unknown")
        events.append(data)
        
        # Print text deltas for visibility
        if event_type == "text_delta":
            print(data.get("text", ""), end="", flush=True)
        elif event_type == "message_update":
            delta = data.get("assistantMessageEvent", {}).get("delta", "")
            if delta:
                print(delta, end="", flush=True)
        
        # Check for tool calls
        if event_type == "toolcall_start":
            print(f"\n[TOOL CALL] {data.get('tool')}")
        elif event_type == "error":
            errors.append(data)
            print(f"\n[ERROR] {data.get('message', 'Unknown error')}")
    
    bridge.set_callback(on_event)
    
    print("=" * 60)
    print("Starting Pi Agent RPC Bridge...")
    print("=" * 60)
    
    try:
        await bridge.start()
        
        # Wait for initialization
        await asyncio.sleep(3)
        
        if not bridge.running:
            print("❌ Bridge failed to start!")
            return False
        
        print("\n✅ Bridge started successfully!")
        print("\n" + "=" * 60)
        print("Sending test prompt...")
        print("=" * 60)
        
        # Send a prompt
        test_prompt = "List the files in the current directory. Use the ls tool."
        print(f"\nPrompt: {test_prompt}\n")
        
        await bridge.send_prompt(test_prompt)
        
        # Wait for response with timeout
        print("Waiting for response (max 600 seconds)...")
        timeout = 600
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(0.5)
            
            # Check for completion events
            done_events = [e for e in events if e.get("type") in ("done", "agent_end", "end")]
            error_events = [e for e in events if e.get("type") == "error"]
            
            if done_events or error_events:
                if done_events:
                    print("\n\n✅ Response complete.")
                break
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        tool_events = [e for e in events if "toolcall" in e.get("type", "")]
        text_events = [e for e in events if e.get("type") == "text_delta"]
        done_events = [e for e in events if e.get("type") == "done"]
        error_events = [e for e in events if e.get("type") == "error"]
        
        print(f"Total events received: {len(events)}")
        print(f"Tool call events: {len(tool_events)}")
        print(f"Text events: {len(text_events)}")
        print(f"Done events: {len(done_events)}")
        print(f"Error events: {len(error_events)}")
        
        if tool_events:
            print("\n✅ SUCCESS: Tools are working!")
            for te in tool_events:
                print(f"   Tool event: {te.get('type')}")
        else:
            print("\n⚠️  No tool events detected - tools may not be enabled")
            
        if error_events:
            print("\n❌ Errors occurred:")
            for ee in error_events:
                print(f"   {ee}")
        
        # Print accumulated text response
        if text_events:
            full_text = "".join([e.get("delta", "") for e in text_events if "delta" in e])
            print(f"\nAgent response preview: {full_text[:500]}...")
        
        return len(tool_events) > 0
        
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\nStopping bridge...")
        await bridge.stop()
        print("Bridge stopped.")


if __name__ == "__main__":
    print("Pi Agent RPC Tool Test")
    print("=" * 60)
    
    result = asyncio.run(test_rpc_with_tools())
    
    print("\n" + "=" * 60)
    if result:
        print("✅ TEST PASSED: Pi Agent tools are working!")
    else:
        print("❌ TEST FAILED: Tools not detected or error occurred")
    print("=" * 60)
    
    sys.exit(0 if result else 1)
