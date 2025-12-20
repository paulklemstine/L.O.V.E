import requests
import os
import json

API_KEY = os.environ.get("STABLE_HORDE", "0000000000")
API_URL = "https://aihorde.net/api/v2/generate/text/async"

def test_payload(name, payload):
    print(f"--- Testing {name} ---")
    headers = {
        "apikey": API_KEY,
        "Content-Type": "application/json",
        "Client-Agent": "LOVE-Agent:v1.0:test"
    }
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

# Base payload matching current code
base_payload = {
    "prompt": "Hello, are you working?",
    "params": {"max_length": 20, "max_context_length": 8192},
    "models": ["Toppy-M-7B"]
}

# 1. Current payload
test_payload("Baseline (Current Code)", base_payload)

# 2. Without max_context_length
payload_no_context = {
    "prompt": "Hello, are you working?",
    "params": {"max_length": 20},
    "models": ["Toppy-M-7B"]
}
test_payload("Without max_context_length", payload_no_context)

# 3. With generic model if Toppy is offline validation issue
payload_generic = {
    "prompt": "Hello, are you working?",
    "params": {"max_length": 20, "max_context_length": 1024},
    "models": [] # Empty list usually picks any available? Or might need specific.
}
# Retrieving a valid model first
try:
    models_resp = requests.get("https://aihorde.net/api/v2/status/models?type=text")
    models_data = models_resp.json()
    if models_data:
        active_models = [m['name'] for m in models_data if m['count'] > 0]
        if active_models:
            print(f"Found active models: {active_models[:3]}")
            payload_generic["models"] = [active_models[0]]
            test_payload(f"With active model {active_models[0]}", payload_generic)
except Exception as e:
    print(f"Failed to fetch models: {e}")
