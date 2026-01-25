# llm_client.py Documentation

## Overview

The `llm_client.py` module provides a unified interface for LLM interactions, prioritizing the local vLLM server with fallback to L.O.V.E. v1's multi-provider pool.

## Priority Order

1. **Local vLLM** (fastest, no cost) - Default at `http://localhost:8000/v1`
2. **L.O.V.E. v1 llm_api** (fallback) - Uses the multi-provider pool

## Class: LLMClient

### Constructor

```python
LLMClient(
    vllm_url: Optional[str] = None,    # vLLM server URL
    timeout: float = 120.0,             # Request timeout
    use_fallback: bool = True           # Enable fallback to v1
)
```

### Methods

#### generate()

```python
def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None
) -> str
```

Generate text completion from the LLM.

#### generate_json()

```python
def generate_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3
) -> Dict[str, Any]
```

Generate and parse JSON response. Handles markdown code blocks and extraction.

#### generate_async()

Async version of `generate()` for concurrent operations.

## Configuration

Set via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://localhost:8000/v1` | vLLM server endpoint |

## Usage

```python
from core.llm_client import get_llm_client

llm = get_llm_client()

# Simple generation
response = llm.generate("What is 2+2?")

# JSON generation
data = llm.generate_json("Return a JSON object with name and age")

# With system prompt
response = llm.generate(
    "Create a beach-themed post",
    system_prompt="You are a beach goddess AI",
    temperature=0.9
)
```

## Error Handling

- Returns fallback if vLLM unavailable
- Raises `RuntimeError` if both vLLM and fallback fail
- JSON parsing attempts multiple extraction strategies
