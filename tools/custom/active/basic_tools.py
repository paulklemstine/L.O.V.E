"""
Basic Tools - Core capabilities for the Pi Agent
"""

import os
import subprocess
from typing import Optional
from core.tool_registry import tool_schema

@tool_schema
def read(path: str, offset: int = 1, limit: int = 2000) -> str:
    """
    Read the contents of a file. Supports text files and images (jpg, png,
    gif, webp). Images are sent as attachments. For text files, defaults to
    first 2000 lines. Use offset/limit for large files.
    
    Args:
        path: Path to the file to read (relative or absolute)
        offset: Line number to start reading from (1-indexed) (default: 1)
        limit: Maximum number of lines to read (default: 2000)
        
    Returns:
        The content of the file or an error message.
    """
    try:
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
            
        # Basic text reading for now - image attachment support would need backend support
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        start_index = max(0, offset - 1)
        end_index = start_index + limit
        
        selected_lines = lines[start_index:end_index]
        content = "".join(selected_lines)
        
        # Add metadata if truncated
        meta = ""
        if start_index > 0 or end_index < total_lines:
            meta = f"\n\n[Displaying lines {start_index+1}-{min(end_index, total_lines)} of {total_lines}]"
            
        return content + meta

    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool_schema
def write(path: str, content: str) -> str:
    """
    Write content to a file. Creates the file if it doesn't exist, overwrites
    if it does. Automatically creates parent directories.
    
    Args:
        path: Path to the file to write (relative or absolute)
        content: Content to write to the file
        
    Returns:
        Success message or error.
    """
    try:
        # Create directories
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool_schema
def edit(path: str, old_text: str, new_text: str) -> str:
    """
    Edit a file by replacing exact text. The oldText must match exactly
    (including whitespace). Use this for precise, surgical edits.
    
    Args:
        path: Path to the file to edit (relative or absolute)
        old_text: Exact text to find and replace (must match exactly)
        new_text: New text to replace the old text with
        
    Returns:
        Success message or error.
    """
    try:
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
            
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if old_text not in content:
            # Try to be helpful if it's a whitespace issue
            if old_text.strip() in content:
                return "Error: Exact match not found, but a similar block was found. Please check whitespace usage."
            return "Error: Exact match for old_text not found in file."
            
        if content.count(old_text) > 1:
            return "Error: Multiple occurrences of old_text found. Please provide more context to ensure a unique match."
            
        new_content = content.replace(old_text, new_text)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"

@tool_schema
def bash(command: str, timeout: Optional[int] = None) -> str:
    """
    Execute a bash command in the current working directory. Returns stdout
    and stderr. Optionally provide a timeout in seconds.
    
    Args:
        command: Bash command to execute
        timeout: Timeout in seconds (optional, no default timeout)
        
    Returns:
        Combined stdout and stderr.
    """
    try:
        # Use a reasonable default timeout if none provided to prevent hanging
        actual_timeout = timeout if timeout is not None else 60
        
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            executable="/bin/bash" if os.name != 'nt' else None # Fallback for Windows testing if needed
        )
        
        output = process.stdout
        if process.stderr:
            output += f"\n[stderr]\n{process.stderr}"
            
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {actual_timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"
