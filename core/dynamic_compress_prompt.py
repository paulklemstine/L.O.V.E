
import inspect
from functools import wraps

def dynamic_compress_prompt(compressor, prompt: str, **kwargs):
    """
    Dynamically compresses a prompt, handling potential TypeError exceptions by
    explicitly removing the 'past_key_values' argument before calling the
    underlying model's forward method.

    Args:
        compressor: The LLMLingua compressor instance.
        prompt (str): The prompt to compress.
        **kwargs: Additional arguments for the compressor.

    Returns:
        The compressed prompt result.
    """
    if not hasattr(compressor, 'model') or not hasattr(compressor.model, 'forward'):
        # Fallback to direct call if the model structure is unexpected
        return compressor.compress_prompt(prompt, **kwargs)

    original_forward = compressor.model.forward

    @wraps(original_forward)
    def patched_forward(*args, **p_kwargs):
        # Explicitly remove the problematic keyword argument
        p_kwargs.pop('past_key_values', None)
        return original_forward(*args, **p_kwargs)

    compressor.model.forward = patched_forward

    try:
        # The underlying call to `compressor.model.forward` will now use the patched version
        result = compressor.compress_prompt(prompt, **kwargs)
    finally:
        # Restore the original method to avoid side effects
        compressor.model.forward = original_forward

    return result
