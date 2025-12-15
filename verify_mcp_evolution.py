import sys
import os
import json
import time
import shutil
from rich.console import Console

# Add current directory to path
sys.path.append(os.getcwd())

from mcp_manager import MCPManager

def test_add_mcp_server():
    console = Console()
    manager = MCPManager(console)
    
    server_name = "test_echo_server"
    
    config = {
        "command": sys.executable,
        "args": ["-c", "import time; print('Mock Server Running'); time.sleep(5)"],
        "env": {"TEST_VAR": "123"}
    }
    
    print(f"Adding server config for '{server_name}'...")
    manager.add_server_config(server_name, config)
    
    # Verify file update
    with open("mcp_servers.json", "r") as f:
        data = json.load(f)
        if server_name in data:
            print("[PASS] mcp_servers.json updated correctly.")
        else:
            print("[FAIL] mcp_servers.json NOT updated.")
            return

    # Verify start
    print(f"Starting server '{server_name}'...")
    result = manager.start_server(server_name)
    print(f"Start result: {result}")
    
    if "successfully" in result:
        print("[PASS] Server started successfully.")
    else:
        print("[FAIL] Server failed to start.")
        
    # Initial cleanup
    manager.stop_server(server_name)
    
if __name__ == "__main__":
    test_add_mcp_server()
