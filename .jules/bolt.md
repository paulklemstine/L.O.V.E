## 2025-02-18 - Replaced O(N) Pixel Access with ImageStat
**Learning:** `list(image.getdata())` is extremely slow (O(N)) and memory-intensive for brightness analysis. `ImageStat.Stat(image)` provides the same metrics (mean, etc.) via C-layer optimization and is ~17x faster for 2048x2048 images.
**Action:** When analyzing image statistics (brightness, contrast), always use `ImageStat` instead of iterating over pixels in Python.
