## 2024-05-23 - [Pixel Access Performance]
**Learning:** `list(image.getdata())` is extremely slow (O(N) in Python) and deprecated. Using `ImageStat` moves the calculation to the C layer, resulting in >10x speedup for brightness calculations.
**Action:** Always check for `getdata()` usage in image processing code and replace with `ImageStat` or `numpy` array operations where possible.
