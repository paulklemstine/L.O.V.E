# bluesky_agent.py Documentation

## Overview

The `bluesky_agent.py` module provides tools for interacting with Bluesky social media, including posting, timeline fetching, replies, and AI-powered content generation.

## Rate Limiting

Built-in cooldown prevents spam:
- **30 minutes** between posts (configurable via `POST_COOLDOWN_SECONDS`)
- Returns error if posting during cooldown

## Functions

### post_to_bluesky()

```python
def post_to_bluesky(
    text: str,
    image_path: Optional[str] = None,
    alt_text: Optional[str] = None
) -> Dict[str, Any]
```

Post to Bluesky with optional image.

**Returns**: `{"success": bool, "post_uri": str|None, "error": str|None}`

**Constraints**:
- Text max 300 characters
- Respects cooldown period

### get_bluesky_timeline()

```python
def get_bluesky_timeline(limit: int = 20) -> Dict[str, Any]
```

Fetch home timeline posts.

**Returns**: `{"success": bool, "posts": List[Dict], "error": str|None}`

### reply_to_post()

```python
def reply_to_post(
    parent_uri: str,
    parent_cid: str,
    text: str
) -> Dict[str, Any]
```

Reply to a specific post.

### search_bluesky()

```python
def search_bluesky(query: str, limit: int = 20) -> Dict[str, Any]
```

Search for posts matching a query.

### generate_post_content()

```python
def generate_post_content(topic: str = None) -> Dict[str, Any]
```

AI-generated post content aligned with persona.

**Returns**: `{"success": bool, "text": str, "hashtags": List[str], "error": str|None}`

## Configuration

Set via `.env`:

| Variable | Description |
|----------|-------------|
| `BLUESKY_USER` | Your Bluesky handle (e.g., `user.bsky.social`) |
| `BLUESKY_PASSWORD` | App password |

## Usage

```python
from core.bluesky_agent import post_to_bluesky, generate_post_content

# Generate content
content = generate_post_content(topic="morning vibes")

if content["success"]:
    # Post it
    result = post_to_bluesky(
        text=content["text"] + " " + " ".join(f"#{t}" for t in content["hashtags"])
    )
    if result["success"]:
        print(f"Posted: {result['post_uri']}")
```

## Integration with PiLoop

These functions are registered as tools via `tool_adapter.py`:
- `bluesky_post` - Posts to Bluesky
- `bluesky_timeline` - Gets timeline
- `bluesky_reply` - Replies to posts
- `bluesky_search` - Searches posts
