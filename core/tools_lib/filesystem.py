
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WriteFileInput(BaseModel):
    filepath: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")
    dry_run: bool = Field(default=False, description="If True, checks compatibility/permissions without writing.")

class ReadFileInput(BaseModel):
    filepath: str = Field(description="The absolute path to the file to read")

@tool("read_file", args_schema=ReadFileInput)
def read_file(filepath: str) -> str:
    """Reads the content of a file."""
    if not filepath:
        return "Error: filepath is required."
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool("write_file", args_schema=WriteFileInput)
def write_file(filepath: str, content: str, dry_run: bool = False) -> str:
    """Writes content to a file."""
    if not filepath or content is None:
        return "Error: filepath and content are required."
    
    # Story 3.2: Dry Run Capability
    if dry_run:
        try:
            if os.path.exists(filepath):
                if os.access(filepath, os.W_OK):
                     return f"Success: File '{filepath}' exists and is writable. Would write {len(content)} bytes."
                else:
                     return f"Error: Permission denied. Cannot write to '{filepath}'."
            else:
                 # Check parent dir
                 parent = os.path.dirname(filepath) or "."
                 if not os.path.exists(parent):
                      return f"Error: Parent directory '{parent}' does not exist."
                 if os.access(parent, os.W_OK):
                      return f"Success: File '{filepath}' does not exist, but parent is writable. Would create file."
                 else:
                      return f"Error: Parent directory '{parent}' is not writable."
        except Exception as e:
            return f"Error during dry run check: {e}"

    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"File '{filepath}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"
