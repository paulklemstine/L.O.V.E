import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
# Need to know the repo. jules_task_manager gets it from git info.
# I'll hardcode it from the previous context "paulklemstine/L.O.V.E" or try to detect.
REPO_OWNER = "paulklemstine"
REPO_NAME = "L.O.V.E"

SESSION_ID = "6344537967799982946"

headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=closed&sort=updated&direction=desc&per_page=10"

print(f"Checking PRs for {REPO_OWNER}/{REPO_NAME}...")
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    prs = response.json()
    
    print(f"Found {len(prs)} closed PRs.")
    for pr in prs:
        merged_at = pr.get("merged_at")
        number = pr.get("number")
        title = pr.get("title")
        body = pr.get("body", "") or ""
        head_ref = pr.get("head", {}).get("ref", "")
        
        print(f"#{number} [{title}] Merged: {merged_at}")
        
        if merged_at:
            # Check for session match
            match = False
            if SESSION_ID in body:
                print("  -> MATCH FOUND in BODY!")
                match = True
            if SESSION_ID in head_ref:
                 print("  -> MATCH FOUND in BRANCH NAME!")
                 match = True
                 
            if match:
                print("  *** THIS IS THE ONE ***")

except Exception as e:
    print(f"Error: {e}")
