## 2025-05-19 - Regex Compilation Performance
**Learning:** Compiling regex patterns in Python via `re.compile()` inside frequently called functions (like `_strip_ansi_codes` in a UI loop) incurs measurable overhead, even with Python's internal caching. Moving these to module-level constants improves performance slightly and is a cleaner pattern.
**Action:** Always define static regex patterns as module-level constants (UPPER_CASE) to avoid repetitive compilation and function call overhead in hot paths.
