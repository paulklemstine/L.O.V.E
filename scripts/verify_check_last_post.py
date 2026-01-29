import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv(str(project_root / ".env"))

from core.bluesky_api import get_own_posts

def check_last_post():
    print("Fetching own posts...")
    posts = get_own_posts(limit=5)
    if not posts:
        print("No posts found.")
        return

    print(f"Found {len(posts)} posts.")
    for i, post in enumerate(posts):
        text = post.value.text.encode('ascii', 'ignore').decode()
        print(f"Post {i}: {text}")
        if "LiveTest" in text:
            print("✅ FOUND LIVE TEST POST!")
            return

    print("❌ Live test post NOT found in recent posts.")

if __name__ == "__main__":
    check_last_post()
