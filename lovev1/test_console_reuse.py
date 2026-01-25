
import io
from rich.console import Console
from rich.text import Text

# Create one console
capture_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor")

# Test 1
capture_console.file = io.StringIO()
capture_console.print(Text("Hello", style="red"))
out1 = capture_console.file.getvalue()
print(f"Out1 length: {len(out1)}")
print(f"Out1 content: {out1!r}")

# Test 2
capture_console.file = io.StringIO()
capture_console.print(Text("World", style="blue"))
out2 = capture_console.file.getvalue()
print(f"Out2 length: {len(out2)}")
print(f"Out2 content: {out2!r}")

# Verify they are different and contain ANSI codes
assert "\x1b" in out1
assert "\x1b" in out2
assert "Hello" in out1
assert "World" in out2
