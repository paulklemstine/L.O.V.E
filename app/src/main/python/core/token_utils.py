import tiktoken
import logging

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
    try:
        # The 'cl100k_base' encoding is used by GPT-3.5 and GPT-4 models.
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback for systems where tiktoken might have issues initializing
        # from a specific encoding.
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logging.warning(f"Could not initialize tiktoken encoder: {e}. Falling back to a rough estimate.")
            # A simple fallback: average of 4 characters per token.
            return len(text) // 4

    return len(encoding.encode(text))