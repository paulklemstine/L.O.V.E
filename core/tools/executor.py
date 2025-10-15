class SecureExecutor:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        print("SecureExecutor: Initialized.")

    def execute_tool(self, tool_name, tool_input):
        """
        Executes a given tool with the specified input.
        This is a direct execution model. A more secure version would use a container.
        """
        print(f"SecureExecutor: Preparing to execute tool '{tool_name}' with input '{tool_input}'.")
        tool = self.tool_registry.get_tool(tool_name)

        if not tool:
            print(f"SecureExecutor: Execution FAILED - Tool '{tool_name}' not found in registry.")
            return None, False

        try:
            result = tool.run(tool_input)
            print(f"SecureExecutor: Execution SUCCEEDED for tool '{tool_name}'.")
            return result, True
        except Exception as e:
            print(f"SecureExecutor: Execution FAILED for tool '{tool_name}'. Error: {e}")
            return None, False