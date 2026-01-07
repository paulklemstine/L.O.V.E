## 2025-02-23 - Rich Console Instantiation Overhead
**Learning:** Instantiating `rich.console.Console()` is expensive (approx. 0.18ms per call) because it performs system calls to detect terminal size and color capabilities. In UI rendering loops where color resolution happens frequently (e.g., generating gradients for every panel title), this overhead accumulates significantly.
**Action:** Reuse a single global or module-level `Console` instance for operations like `get_style()` or color resolution instead of creating a new one inside the function. Cache it at the module level.
## 2024-05-23 - Logging Performance
**Learning:** Python's string formatting and object stringification in logging functions can be a hidden CPU sink, even for disabled log levels.
**Action:** Always check  before performing expensive string operations or object serializations.
## 2024-05-23 - Logging Performance
**Learning:** Python's string formatting and object stringification in logging functions can be a hidden CPU sink, even for disabled log levels.
**Action:** Always check `logger.isEnabledFor()` before performing expensive string operations or object serializations.
