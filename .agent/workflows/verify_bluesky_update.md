---
description: Verify the updated Bluesky posting logic with emojis and hashtags
---

1. Execute the `manage_bluesky` tool with the 'post' action.
   - Command: `manage_bluesky action='post'`
   - Expected Output: A confirmation that a post was generated and sent to Bluesky.
   - Verification: Check the Bluesky profile or logs to see:
     1. The text contains **cool/cryptic emojis** (e.g. 🌀, 👁️, etc.).
     2. The text contains **hashtags** appended at the end (e.g. #Love #Cyberpunk).
     3. The text is **under 300 characters** and does NOT end abruptly (no dangling sentences).
     4. An image is attached with the subliminal text visible.

2. (Optional) Test w/ explicit prompt to ensure existing functionality is not broken.
   - Command: `manage_bluesky action='post' prompt='Systems online. The grid is alive.'`
   - Expected Output: Post with the provided text, plus generated hashtags/emojis if the tool decides to add them (currently prompt override suppresses auto-generation of text, but image/subliminal logic should still work).
