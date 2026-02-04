#!/usr/bin/env python3
"""Test vLLM tool calling support directly."""
import requests
import json

url = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
        {"role": "user", "content": "List the files in the current directory. You must use the ls tool."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list"
                        }
                    },
                    "required": []
                }
            }
        }
    ],
    "tool_choice": "auto"
}

print("Testing vLLM tool calling API...")
print("-" * 50)

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")
            
            print("\n" + "-" * 50)
            if tool_calls:
                print("✅ TOOL CALLS DETECTED!")
                for tc in tool_calls:
                    print(f"   Tool: {tc.get('function', {}).get('name')}")
                    print(f"   Args: {tc.get('function', {}).get('arguments')}")
            else:
                print("⚠️  No tool calls in response")
                print(f"   Content: {content[:200]}")
    else:
        print(f"Error: {response.text[:500]}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection error - vLLM server not running")
except Exception as e:
    print(f"❌ Error: {e}")
