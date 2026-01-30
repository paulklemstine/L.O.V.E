## 2024-05-23 - Pillow Performance & Deprecation
**Learning:** `list(image.getdata())` converts image data to a Python list, which is O(N) and memory intensive. It is also deprecated in newer Pillow versions.
**Action:** Always use `ImageStat` for statistical analysis of images (mean, median, etc.) as it runs in the C layer and is significantly faster (~10x).
