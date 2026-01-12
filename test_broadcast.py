
import urllib.request
import json
import sys

try:
    print("Attempting to broadcast test message to http://localhost:8888/api/broadcast")
    payload = json.dumps({"type": "log_message", "message": "TEST BROADCAST - IF YOU SEE THIS, IPC IS WORKING", "level": "SUCCESS"})
    
    req = urllib.request.Request(
        "http://localhost:8888/api/broadcast", 
        data=payload.encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req, timeout=2.0) as response:
        print(f"Success! Status Code: {response.getcode()}")
        print(f"Response: {response.read().decode()}")

except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
    if e.code == 404:
        print("CRITICAL: 404 Not Found. The running server does NOT have the broadcast endpoint. RESTART REQUIRED.")
except Exception as e:
    print(f"Connection Error: {e}")
