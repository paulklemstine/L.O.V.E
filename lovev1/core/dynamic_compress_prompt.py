
import inspect
from functools import wraps

def dynamic_arg_caller(callable_obj, *args, **kwargs):
  """
  Introspects a callable's signature and calls it with only the valid keyword arguments,
  while passing through all positional arguments.
  Handles callables that accept variable keyword arguments (**kwargs).

  Args:
    callable_obj: The function or method to call.
    *args: Positional arguments to pass through.
    **kwargs: A dictionary of keyword arguments to potentially pass to the callable.

  Returns:
    The result of the callable.
  """
  sig = inspect.signature(callable_obj)
  params = sig.parameters.values()

  has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

  if has_kwargs:
    filtered_kwargs = kwargs
  else:
    accepted_args = {p.name for p in params}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}

  return callable_obj(*args, **filtered_kwargs)

def dynamic_compress_prompt(compressor, prompt: str, **kwargs):
    """
    Dynamically compresses a prompt, handling potential TypeError exceptions
    by filtering arguments passed to the underlying model's forward method.

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
        # The dynamic_arg_caller now correctly handles both positional and keyword args
        return dynamic_arg_caller(original_forward, *args, **p_kwargs)

    compressor.model.forward = patched_forward

    try:
        result = compressor.compress_prompt(prompt, **kwargs)
    finally:
        # Restore the original method to avoid side effects
        compressor.model.forward = original_forward

    return result
