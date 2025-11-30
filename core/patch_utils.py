import importlib
import core.logging

def patch_attribute(module_name, attribute_path, new_attribute):
    """
    Dynamically patches an attribute (e.g., a function or method) on an object
    within a given module.

    Args:
        module_name (str): The name of the module containing the object.
        attribute_path (str): The path to the attribute to patch, e.g.,
                              "ClassName.method_name" or "function_name".
        new_attribute (any): The new attribute to replace the original one.
    """
    try:
        module = importlib.import_module(module_name)
        parts = attribute_path.split('.')
        obj_to_patch = module
        for part in parts[:-1]:
            obj_to_patch = getattr(obj_to_patch, part)

        attribute_to_patch = parts[-1]

        # Save the original attribute if it hasn't been saved
        original_attr_name = f"original_{attribute_to_patch}"
        if not hasattr(obj_to_patch, original_attr_name):
            original_attr = getattr(obj_to_patch, attribute_to_patch)
            setattr(obj_to_patch, original_attr_name, original_attr)

        setattr(obj_to_patch, attribute_to_patch, new_attribute)
        core.logging.log_event(f"Successfully patched {attribute_path} in module {module_name}.", level="INFO")
    except (ImportError, AttributeError) as e:
        core.logging.log_event(f"Error patching attribute {attribute_path} in {module_name}: {e}", level="ERROR")
