import os
import requests
import json
import sys
from dotenv import load_dotenv

load_dotenv()

SESSION_NAME = "sessions/6344537967799982946"
API_KEY = os.environ.get("JULES_API_KEY")

if not API_KEY:
    print("Error: JULES_API_KEY not found in environment.")
    sys.exit(1)

url = f"https://jules.googleapis.com/v1alpha/{SESSION_NAME}/activities"
headers = {"Content-Type": "application/json", "X-Goog-Api-Key": API_KEY}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Sort activities
    activities = data.get("activities", [])
    activities.sort(key=lambda x: x.get("createTime", ""))
    
    print(f"\nFound {len(activities)} activities.")
    
    last_seen_id = "3e69585fbf854aaf9cec384e26b7271f"
    seen_last = False

    for activity in activities:
        act_id = activity.get("name", "").split("/")[-1]
        act_type = activity.get("type", "unknown")
        detail_keys = list(activity.get("detail", {}).keys())
        
        prefix = "   "
        if act_id == last_seen_id:
            prefix = ">>>"
            seen_last = True
        elif seen_last:
            prefix = "NEW"
            
        print(f"{prefix} [{activity.get('createTime')}] ID: {act_id} Type: {act_type} Details: {detail_keys}")
        
        if "pullRequest" in detail_keys:
             url = activity["detail"]["pullRequest"].get("url")
             print(f"    -> PR URL: {url}")

            
except Exception as e:
    print(f"Error fetching activities: {e}")
