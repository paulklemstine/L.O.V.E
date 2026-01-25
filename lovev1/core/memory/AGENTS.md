# AGENTS.md - Memory Systems

## Purpose
Manages long-term and short-term memory systems, including Vector DBs (FAISS) and semantic search.

## Invariants
- **Persistence**: Memory changes should be persisted to disk (e.g., `faiss_index.bin`) periodically or immediately.
- **Recall**: Just saving is not enough. Ensure there are effective retrieval mechanisms.

## Anti-patterns
- **Memory Leaks**: Loading the entire vector index into RAM if it grows too large (monitor usage).
- **Orphaned Embeddings**: Deleting a record but leaving its embedding in the index.
