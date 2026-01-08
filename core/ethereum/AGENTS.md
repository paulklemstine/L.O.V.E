# AGENTS.md - Ethereum & Blockchain

## Purpose
Handles all blockchain interactions, wallet management, and smart contract calls.

## Critical Invariants
> [!CRITICAL]
> **Safety First**: Never sign a transaction without `SecureTransactionManager` validation. Use `secure_executor.py` where possible.

## Architecture
- `simulator.py`: Used for dry-runs and predicting gas usage. Safe to call freely.
- `transaction.py` / `wallet`: ACTUAL execution. Requires unlocked keys and high confidence.

## Anti-patterns
- Hardcoding private keys (Use `os.getenv`).
- Sending transactions in a tight loop without waiting for confirmation.
