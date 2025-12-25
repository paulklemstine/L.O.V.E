## 2024-05-22 - [Double Logging Overhead]
**Learning:** The logging system had a severe inefficiency where `log_event` in `core/logging.py` was queueing ALL log messages to the UI queue, regardless of the configured log level. Furthermore, `simple_ui_renderer` in `love.py` was writing these messages to `love.log` *again*, resulting in duplicate I/O and disk usage.
**Action:** Always check if a log level is enabled before doing work (queueing/formatting). Ensure single responsibility for file I/O to avoid duplication.

## 2024-05-23 - [UI Render Loop Optimization]
**Learning:** The UI render loop (`simple_ui_renderer`) was repeatedly compiling a regex and opening/closing the log file for every single frame update. This caused significant syscall overhead.
**Action:** Pre-compile regexes at the module level and keep file handles open for long-running loops where possible, ensuring `flush()` is called to maintain real-time logging.
