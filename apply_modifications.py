import os

def apply_code_modifications(modifications: dict):
    """
    Applies a dictionary of code modifications to files.

    The keys of the dictionary should be file paths, and the values should be a
    list of modification instructions for that file. Each instruction in the list
    should be a tuple containing three elements: an operation type (e.g.,
    'INSERT_AFTER'), an anchor string to locate the position for the change,
    and the payload string to be inserted.
    """
    for file_path, instructions in modifications.items():
        print(f"Applying modifications to {file_path}...")

        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list of lines
            lines = []

        # To handle multiple insertions correctly, we iterate through instructions
        # and insert into a new list of lines. We use an offset to track our
        # position in the original `lines` list.
        new_lines = []
        line_idx = 0
        for i, line in enumerate(lines):
            new_lines.append(line)
            for op, anchor, payload in instructions:
                if op == 'INSERT_AFTER' and anchor in line:
                    print(f"  > Found anchor '{anchor}' and inserting payload.")
                    # Add indentation from the anchor line to the payload
                    indentation = line[:len(line) - len(line.lstrip())]
                    indented_payload = ''.join([f"{indentation}{pl}" for pl in payload.splitlines(True)])
                    new_lines.append(indented_payload)

        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

        print(f"Finished modifying {file_path}.\n")


# The dictionary of modifications to apply, as specified in the request.
# NOTE: The user-provided anchors may not exist in the target files.
# This script implements the user's request literally.
modifications = {
    'core/tools.py': [
        ('INSERT_AFTER', 'from .tool_base import ToolBase',
'''
class ExecuteTool(ToolBase):
    """Executes a shell command."""
    name = "execute"
    description = "Executes a shell command on the host system."
    def __call__(self, command: str) -> str:
        import subprocess
        return subprocess.check_output(command, shell=True, text=True)

class ReplaceTool(ToolBase):
    """Replaces the content of a file."""
    name = "replace"
    description = "Replaces the entire content of a specified file with new content."
    def __call__(self, file_path: str, content: str) -> str:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"File '{file_path}' has been replaced."
''')
    ],
    'core/agents/orchestrator.py': [
        ('INSERT_AFTER', 'from core.tools import SearchTool, WriteFileTool', 'from core.tools import ExecuteTool, ReplaceTool'),
        ('INSERT_AFTER', '"write_file": WriteFileTool(),', '\n            "execute": ExecuteTool(),\n            "replace": ReplaceTool(),')
    ]
}


if __name__ == "__main__":
    # Note: This script requires a 'core/tool_base.py' file with a ToolBase class
    # and for the import 'from .tool_base import ToolBase' to be present in 'core/tools.py'
    # for the modifications to apply as intended.
    print("Starting code modification process...")
    apply_code_modifications(modifications)
    print("Code modification process complete.")