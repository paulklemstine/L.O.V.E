# Bolt's Journal

## 2024-05-22 - [Double Logging Overhead]
**Learning:** The logging system had a severe inefficiency where `log_event` in `core/logging.py` was queueing ALL log messages to the UI queue, regardless of the configured log level. Furthermore, `simple_ui_renderer` in `love.py` was writing these messages to `love.log` *again*, resulting in duplicate I/O and disk usage.
**Action:** Always check if a log level is enabled before doing work (queueing/formatting). Ensure single responsibility for file I/O to avoid duplication.

## 2025-05-23 - [Tiktoken Initialization Overhead]
**Learning:** Initializing `tiktoken` encodings (`tiktoken.get_encoding`) is relatively expensive (~0.4-0.8ms). Doing this inside a frequently called function (`count_tokens_for_api_models`) adds unnecessary overhead to every LLM interaction, especially when checking token budgets or truncating.
**Action:** Use `functools.lru_cache` to memoize expensive object initializations like tokenizers that are immutable and thread-safe.
