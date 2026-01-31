## 2025-02-18 - Pillow ImageStat vs getdata()
**Learning:** `list(image.getdata())` is extremely slow (O(N) in Python) for pixel access.
**Action:** Always use `ImageStat.Stat` (C-layer) for aggregate statistics like brightness or average color. It's ~10x faster.
