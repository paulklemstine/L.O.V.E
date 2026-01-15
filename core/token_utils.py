import tiktoken
import logging
from functools import lru_cache
from typing import Optional

# Global cache for the tokenizer encoding to avoid repeated initialization/lookup overhead
_cached_encoding: Optional[tiktoken.Encoding] = None

def _get_encoding(model_name: str = "gpt-4") -> Optional[tiktoken.Encoding]:
    """
    Retrieves the tiktoken encoding, using a global cache to maximize performance.
    """
    global _cached_encoding
    if _cached_encoding is not None:
        return _cached_encoding

    try:
        # The 'cl100k_base' encoding is used by GPT-3.5 and GPT-4 models.
        _cached_encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback for systems where tiktoken might have issues initializing
        # from a specific encoding.
        try:
            _cached_encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logging.warning(f"Could not initialize tiktoken encoder: {e}. Falling back to a rough estimate.")
            return None

    return _cached_encoding

@lru_cache(maxsize=128)
def count_tokens_for_api_models(text: str, model_name: str = "gpt-4") -> int:
    """
    Estimates the number of tokens in a string of text for API-based models.

    This function uses tiktoken, which is the tokenizer used by OpenAI's models.
    It serves as a good general-purpose tokenizer for many modern LLMs,
    including the Gemini and other API-based models used in this project.

    Optimized with LRU cache to handle repeated calls for the same text (e.g. logging vs stats).

    Args:
        text: The string to be tokenized.
        model_name: The name of the model to use for tokenization.
                    Defaults to "gpt-4" as a robust choice.

    Returns:
        The estimated number of tokens.
    """
    if not text:
        return 0

    encoding = _get_encoding(model_name)

    if encoding:
        return len(encoding.encode(text))
    else:
        # A simple fallback: average of 4 characters per token.
        return len(text) // 4
