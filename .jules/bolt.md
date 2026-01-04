## 2025-02-23 - Rich Console Instantiation Overhead
**Learning:** Instantiating `rich.console.Console()` is expensive (approx. 0.18ms per call) because it performs system calls to detect terminal size and color capabilities. In UI rendering loops where color resolution happens frequently (e.g., generating gradients for every panel title), this overhead accumulates significantly.
**Action:** Reuse a single global or module-level `Console` instance for operations like `get_style()` or color resolution instead of creating a new one inside the function. Cache it at the module level.
