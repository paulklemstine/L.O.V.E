import io
from rich.console import Console

class OffscreenRenderer:
    """
    Renders Rich content to a string buffer instead of stdout.
    Used for creating string representations of panels for logs or network transmission.
    """
    def __init__(self, width=80):
        self.width = width
        # force_terminal=True ensures ANSI codes are generated
        self.console = Console(file=io.StringIO(), width=width, force_terminal=True, color_system="truecolor", highlight=False)

    def render(self, renderable, width=None):
        if width:
            self.console.width = width
        
        # Reset the buffer
        self.console.file = io.StringIO()
        self.console.print(renderable)
        return self.console.file.getvalue()
