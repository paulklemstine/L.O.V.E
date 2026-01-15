import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.google_auth import get_jules_access_token, get_jules_auth_headers, initialize_google_auth
import requests

def test_jules():
    # Initialize auth (this will check gcloud, etc.)
    success, message = initialize_google_auth()
    print(f"Auth init: {message}")
    
    # Get headers (includes token + quota project)
    headers = get_jules_auth_headers()
    if not headers:
        print("ERROR: No valid OAuth token available")
        print("Please run: gcloud auth login")
        return

    headers["Content-Type"] = "application/json"
    print(f"Headers: {list(headers.keys())}")
    
    # Test the specific session that failed
    session_name = "sessions/1257419563107173607"
    url = f"https://jules.googleapis.com/v1alpha/{session_name}/activities"
    
    print(f"Testing URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text[:500]}")
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_jules()
