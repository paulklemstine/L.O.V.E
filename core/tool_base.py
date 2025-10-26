class ToolBase:
    """Base class for all tools."""
    name = "tool_base"
    description = "This is a base class and should not be used directly."

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")