import logging
from functools import lru_cache
import tiktoken


@lru_cache(maxsize=1)
def _get_cached_encoding(encoding_name: str):
    """
    Cached retrieval of tiktoken encoding to avoid repeated overhead.
    """
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=1)
def _get_cached_encoding_for_model(model_name: str):
    """
    Cached retrieval of tiktoken encoding for a specific model.
    """
    return tiktoken.encoding_for_model(model_name)


def count_tokens_for_api_models(text: str, model_name: str = "gpt-4") -> int:
    """
    Estimates the number of tokens in a string of text for API-based models.

    This function uses tiktoken, which is the tokenizer used by OpenAI's models.
    It serves as a good general-purpose tokenizer for many modern LLMs,
    including the Gemini and other API-based models used in this project.

    Args:
        text: The string to be tokenized.
        model_name: The name of the model to use for tokenization.
                    Defaults to "gpt-4" as a robust choice.

    Returns:
        The estimated number of tokens.
    """
    if not text:
        return 0

    encoding = None
    try:
        # The 'cl100k_base' encoding is used by GPT-3.5 and GPT-4 models.
        # Use cached function to avoid overhead of creating encoding object every time
        encoding = _get_cached_encoding("cl100k_base")
    except Exception:  # pylint: disable=broad-exception-caught
        # Fallback for systems where tiktoken might have issues initializing
        # from a specific encoding.
        try:
            encoding = _get_cached_encoding_for_model(model_name)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Could not initialize tiktoken encoder: %s. Falling back to a rough estimate.", e)
            # A simple fallback: average of 4 characters per token.
            return len(text) // 4

    return len(encoding.encode(text))
