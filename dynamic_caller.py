
import inspect

def dynamic_arg_caller(callable_obj, **kwargs):
  """
  Introspects a callable's signature and calls it with only the valid keyword arguments.
  Handles callables that accept variable keyword arguments (**kwargs).

  Args:
    callable_obj: The function or method to call.
    **kwargs: A dictionary of keyword arguments to potentially pass to the callable.

  Returns:
    The result of the callable.
  """
  # Get the signature of the callable
  sig = inspect.signature(callable_obj)
  params = sig.parameters.values()

  # Check if the callable accepts variable keyword arguments (**kwargs)
  has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

  if has_kwargs:
    # If it accepts **kwargs, pass all arguments through
    filtered_kwargs = kwargs
  else:
    # Otherwise, filter to only include accepted argument names
    accepted_args = {p.name for p in params}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}

  # Call the callable with the appropriate arguments
  return callable_obj(**filtered_kwargs)

# --- Demonstration ---

class MockXLMRoberta:
  """A mock class to simulate the XLMRobertaForTokenClassification model."""
  def forward(self, input_ids, attention_mask=None, labels=None):
    """
    A mock forward method that accepts specific arguments.
    """
    print("MockXLMRoberta.forward called successfully!")
    print(f"  - input_ids: {input_ids}")
    print(f"  - attention_mask: {attention_mask}")
    print(f"  - labels: {labels}")
    return "SUCCESS"

if __name__ == "__main__":
  # 1. Setup the scenario
  model = MockXLMRoberta()

  # Define a set of arguments, including an unsupported one
  all_arguments = {
      'input_ids': [1, 2, 3],
      'attention_mask': [1, 1, 0],
      'labels': ['B-PERSON', 'I-PERSON', 'O'],
      'past_key_val': 'some_value'  # This argument is not supported by `forward`
  }

  print("--- Scenario ---")
  print(f"Calling a method with these arguments: {list(all_arguments.keys())}")
  print("-" * 20)

  # 2. Demonstrate the direct call failure
  print("\nAttempting a direct call (expected to fail)...")
  try:
    model.forward(**all_arguments)
  except TypeError as e:
    print(f"  -> Caught expected error: {e}")

  # 3. Demonstrate the call using the dynamic_arg_caller
  print("\nAttempting call with `dynamic_arg_caller`...")
  result = dynamic_arg_caller(model.forward, **all_arguments)
  print(f"\n  -> Result from dynamic_arg_caller: {result}")

  print("\n" + "="*40 + "\n")

  # --- New Demonstration for **kwargs ---

  def function_with_kwargs(required_arg, **kwargs):
    """A function that accepts a required argument and **kwargs."""
    print("function_with_kwargs called successfully!")
    print(f"  - required_arg: {required_arg}")
    print(f"  - other kwargs: {kwargs}")
    return "KWAGRS_SUCCESS"

  kwargs_arguments = {
      'required_arg': 'hello',
      'extra_arg_1': 100,
      'extra_arg_2': True
  }

  print("--- Scenario with **kwargs ---")
  print(f"Calling a function with these arguments: {list(kwargs_arguments.keys())}")
  print("-" * 20)

  print("\nAttempting call with `dynamic_arg_caller`...")
  result_kwargs = dynamic_arg_caller(function_with_kwargs, **kwargs_arguments)
  print(f"\n  -> Result from dynamic_arg_caller: {result_kwargs}")
