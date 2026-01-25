import os
import importlib.util
import sys
from typing import List

class ExtensionsManager:
    def __init__(self, extensions_dir: str = "extensions"):
        # Helper to find project root or relative to core
        # If this file is in core/, .. is root.
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.extensions_dir = os.path.join(root_dir, extensions_dir)
        
    def load_extensions(self, tool_registry):
        """
        Scans the extensions directory for Python scripts and loads them.
        Expects scripts to have a 'register_tools(registry)' function.
        """
        if not os.path.exists(self.extensions_dir):
             try:
                os.makedirs(self.extensions_dir)
             except OSError:
                pass
             return

        # print(f"Scanning extensions in {self.extensions_dir}...")
        for filename in os.listdir(self.extensions_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                file_path = os.path.join(self.extensions_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Avoid conflict with existing modules by using specific name if needed
                        # sys.modules[f"extensions.{module_name}"] = module 
                        spec.loader.exec_module(module)
                        
                        if hasattr(module, "register_tools"):
                            # print(f"Loading extension: {module_name}")
                            module.register_tools(tool_registry)
                        # else:
                        #     print(f"Extension {module_name} missing 'register_tools' function. Skipping.")
                except Exception as e:
                    print(f"Failed to load extension {module_name}: {e}")
