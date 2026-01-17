
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import core.shared_state as shared_state

class ExecuteInput(BaseModel):
    command: str = Field(description="The shell command to run")

@tool("execute", args_schema=ExecuteInput)
async def execute(command: str) -> str:
    """Executes a shell command."""
    if not command:
        return "Error: The 'execute' tool requires a 'command' argument. Please specify the shell command to execute."
    
    from network import execute_shell_command
    
    # We pass love_state if available, else empty dict
    state = getattr(shared_state, 'love_state', {})
    
    return str(execute_shell_command(command, state))
