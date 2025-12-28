import os
import requests

def load_env():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip('"')

def test_jules():
    load_env()
    api_key = os.environ.get("JULES_API_KEY")
    if not api_key:
        print("ERROR: JULES_API_KEY not found in .env")
        return

    print(f"Loaded JULES_API_KEY: {api_key[:5]}...{api_key[-5:]}")
    
    # Test the specific session that failed
    session_name = "sessions/1257419563107173607"
    url = f"https://jules.googleapis.com/v1alpha/{session_name}/activities"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
    
    print(f"Testing URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_jules()
