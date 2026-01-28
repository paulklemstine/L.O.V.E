## 2024-05-23 - [Pillow ImageStat vs getdata()]
**Learning:** `list(image.getdata())` creates a full Python list of pixel values, which is extremely slow and memory-intensive (O(N) memory and Python loop overhead). `ImageStat.Stat(image).mean` uses optimized C implementation (O(1) Python overhead).
**Action:** Always use `ImageStat` or histogram methods for global image statistics instead of iterating over pixels in Python.
