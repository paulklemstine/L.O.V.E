## 2024-05-22 - [Pillow Image Analysis Performance]
**Learning:** `list(image.getdata())` is extremely slow and memory-intensive for pixel analysis, and it is deprecated. `ImageStat.Stat(image)` is implemented in C and is orders of magnitude faster (~25x for 1024x1024).
**Action:** Always prefer `ImageStat` or `numpy` (if available) for image statistics over iterating pixels in Python.
