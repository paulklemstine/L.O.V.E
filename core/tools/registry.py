class ToolRegistry:
    def __init__(self):
        self.tools = {}
        print("ToolRegistry: Initialized.")

    def register_tool(self, tool_instance):
        """Registers a tool instance."""
        tool_name = tool_instance.name
        self.tools[tool_name] = tool_instance
        print(f"ToolRegistry: Registered tool '{tool_name}'.")

    def get_tool(self, tool_name):
        """Retrieves a tool by its name."""
        return self.tools.get(tool_name)