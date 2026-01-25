#!/usr/bin/env python3
"""
Test script for L.O.V.E. Live API.

Usage:
    python tests/test_live_api.py [--api-key YOUR_KEY] [--base-url http://localhost:8888]
"""

import argparse
import json
import sys
import asyncio

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


async def test_live_api(base_url: str, api_key: str):
    """Run API tests."""
    
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        headers = {"X-API-Key": api_key} if api_key else {}
        
        print("=" * 60)
        print("L.O.V.E. Live API Test Suite")
        print("=" * 60)
        print(f"Base URL: {base_url}")
        print(f"API Key: {'*' * 20 if api_key else 'None (using auto-generated)'}")
        print()
        
        # Test 1: Health check (no auth required)
        print("[1] Testing /api/health (no auth)...")
        try:
            response = await client.get("/api/health")
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Health OK: {data.get('status')}")
                if "components" in data:
                    print(f"    Components: {list(data['components'].keys())}")
            else:
                print(f"    ❌ Unexpected status: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        # Test 2: State (requires auth)
        print("[2] Testing /api/state (auth required)...")
        try:
            response = await client.get("/api/state", headers=headers)
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ State retrieved: {len(data)} keys")
                if "_meta" in data:
                    print(f"    Meta: {data['_meta']}")
            elif response.status_code == 401:
                print("    ⚠️ Unauthorized - check API key")
            else:
                print(f"    ❌ Error: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        # Test 3: List tools
        print("[3] Testing /api/tools...")
        try:
            response = await client.get("/api/tools", headers=headers)
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                tool_count = data.get("count", 0)
                print(f"    ✅ Tools found: {tool_count}")
                if data.get("tools"):
                    sample = data["tools"][:3]
                    for t in sample:
                        print(f"       - {t['name']}: {t.get('description', '')[:50]}...")
            else:
                print(f"    ❌ Error: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        # Test 4: Agent status
        print("[4] Testing /api/agent/status...")
        try:
            response = await client.get("/api/agent/status", headers=headers)
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Agent active: {data.get('agent_active')}")
                print(f"    Completed tasks: {data.get('completed_tasks_count', 0)}")
            else:
                print(f"    ❌ Error: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        # Test 5: Hooks list
        print("[5] Testing /api/hooks...")
        try:
            response = await client.get("/api/hooks", headers=headers)
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Hooks stats: {data.get('stats')}")
            else:
                print(f"    ❌ Error: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        # Test 6: Submit command
        print("[6] Testing /api/command...")
        try:
            response = await client.post(
                "/api/command",
                headers=headers,
                json={"command": "Test command from API", "async": True}
            )
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Command status: {data.get('status')}")
            else:
                print(f"    ❌ Error: {response.text}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        print()
        
        print("=" * 60)
        print("Test suite complete!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test L.O.V.E. Live API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8888",
        help="Base URL for the API (default: http://localhost:8888)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication"
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_live_api(args.base_url, args.api_key))


if __name__ == "__main__":
    main()
