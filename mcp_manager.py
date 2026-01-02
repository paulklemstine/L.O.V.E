# mcp_manager.py

import subprocess
import threading
import json
import queue
import os
import time
import uuid

import core.logging

class MCPManager:
    """
    Manages the lifecycle and communication with local MCP server subprocesses.
    """
    def __init__(self, console):
        self.console = console
        self.servers = {}
        self.lock = threading.Lock()
        self.server_configs = self._load_server_configs()

    def _load_server_configs(self):
        """Loads server definitions from mcp_servers.json."""
        config_path = "mcp_servers.json"
        if not os.path.exists(config_path):
            core.logging.log_event(f"MCP config file not found at '{config_path}'. No servers can be started.", "WARNING")
            return {}
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            core.logging.log_event(f"Error loading MCP config file '{config_path}': {e}", "ERROR")
            return {}

    def _read_output(self, pipe, q):
        """Reads lines from a subprocess pipe and puts them into a queue."""
        try:
            for line in iter(pipe.readline, ''):
                q.put(line)
        finally:
            pipe.close()

    def check_missing_env_vars(self, server_name):
        """
        Checks for missing environment variables required by a server.
        Returns a list of missing environment variable names.
        """
        missing_vars = []
        config = self.server_configs.get(server_name)
        if config:
            required_vars = config.get("requires_env", [])
            for var in required_vars:
                if var not in os.environ:
                    missing_vars.append(var)
        return missing_vars

    def start_server(self, server_name, env_vars=None):
        """
        Starts a defined MCP server as a subprocess.
        `env_vars` is a dictionary of environment variables to pass to the subprocess.
        """
        with self.lock:
            if server_name in self.servers and self.servers[server_name]['process'].poll() is None:
                return f"Server '{server_name}' is already running."

            missing_vars = self.check_missing_env_vars(server_name)
            if missing_vars:
                var_list = ", ".join(missing_vars)
                return (
                    f"Error: Cannot start MCP server '{server_name}'. "
                    f"The following required environment variables are missing: {var_list}. "
                    "Please set them before starting the server."
                )

            config = self.server_configs.get(server_name)
            if not config:
                return f"Error: Server '{server_name}' not found in mcp_servers.json."

            try:
                command = config['command']
                args = config.get('args', [])
                full_command = [command] + args

                # Combine provided env_vars with the current environment
                process_env = os.environ.copy()
                if env_vars:
                    process_env.update(env_vars)

                process = subprocess.Popen(
                    full_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=process_env,
                    bufsize=1 # Line-buffered
                )

                response_queue = queue.Queue()
                stderr_queue = queue.Queue()

                stdout_thread = threading.Thread(target=self._read_output, args=(process.stdout, response_queue), daemon=True)
                stderr_thread = threading.Thread(target=self._read_output, args=(process.stderr, stderr_queue), daemon=True)
                stdout_thread.start()
                stderr_thread.start()

                self.servers[server_name] = {
                    "process": process,
                    "config": config,
                    "response_queue": response_queue,
                    "stderr_queue": stderr_queue,
                    "stdout_thread": stdout_thread,
                    "stderr_thread": stderr_thread,
                    "pending_requests": {}
                }
                core.logging.log_event(f"MCP server '{server_name}' started with PID {process.pid}.", "INFO")
                return f"Server '{server_name}' started successfully."

            except (FileNotFoundError, subprocess.SubprocessError) as e:
                core.logging.log_event(f"Failed to start MCP server '{server_name}': {e}", "CRITICAL")
                return f"Error: Failed to start server '{server_name}': {e}"

    def stop_server(self, server_name):
        """Stops a running MCP server."""
        with self.lock:
            if server_name not in self.servers or self.servers[server_name]['process'].poll() is not None:
                return f"Server '{server_name}' is not running."

            server_info = self.servers[server_name]
            process = server_info['process']

            try:
                process.terminate()
                process.wait(timeout=10)
                core.logging.log_event(f"MCP server '{server_name}' terminated.", "INFO")
            except subprocess.TimeoutExpired:
                process.kill()
                core.logging.log_event(f"MCP server '{server_name}' killed forcefully.", "WARNING")

            del self.servers[server_name]
            return f"Server '{server_name}' stopped."

    def list_running_servers(self):
        """Returns a list of running MCP servers."""
        with self.lock:
            running = []
            for name, info in self.servers.items():
                if info['process'].poll() is None:
                    running.append({"name": name, "pid": info['process'].pid})
            return running

    def list_tools(self, server_name: str) -> dict:
        """
        Returns the tool definitions for a server from its configuration.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Dictionary mapping tool names to their descriptions
        """
        config = self.server_configs.get(server_name, {})
        return config.get("tools", {})


    def call_tool(self, server_name, tool_name, params):
        """
        Calls a tool on a running MCP server using JSON-RPC.
        Returns a request_id to check for the response later.
        """
        with self.lock:
            if server_name not in self.servers or self.servers[server_name]['process'].poll() is not None:
                raise ValueError(f"Server '{server_name}' is not running.")

            server_info = self.servers[server_name]
            request_id = str(uuid.uuid4())

            # MCP uses a specific JSON structure
            # Note: MCP/v1 specifies 'jsonrpc':'2.0', id, method, and params.
            # To call a tool, the method is 'tools/call' and the tool name is passed in 'params'.
            mcp_request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }

            try:
                request_str = json.dumps(mcp_request) + '\n'
                server_info['process'].stdin.write(request_str)
                server_info['process'].stdin.flush()

                # Store the request so we can match the response later
                server_info['pending_requests'][request_id] = time.time()

                core.logging.log_event(f"Sent request {request_id} to '{server_name}': {tool_name}", "INFO")
                return request_id

            except (IOError, BrokenPipeError) as e:
                core.logging.log_event(f"Error communicating with server '{server_name}': {e}", "ERROR")
                raise IOError(f"Failed to send request to '{server_name}'. It may have crashed.") from e

    def get_response(self, server_name, request_id, timeout=30):
        """
        Retrieves the response for a specific request ID, waiting if necessary.
        """
        with self.lock:
            if server_name not in self.servers:
                return {"error": {"message": f"Server '{server_name}' not found."}}

            server_info = self.servers[server_name]
            if request_id not in server_info['pending_requests']:
                return {"error": {"message": f"Request ID '{request_id}' not pending for this server."}}

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response_line = server_info['response_queue'].get_nowait()
                try:
                    response_json = json.loads(response_line)
                    if response_json.get('id') == request_id:
                        with self.lock:
                            del server_info['pending_requests'][request_id]
                        return response_json
                    else:
                        # It's a response for a different request, put it back for now.
                        # This is a simplification; a better system might have a dict of queues.
                        server_info['response_queue'].put(response_line)
                except json.JSONDecodeError:
                    # Ignore non-JSON lines which might be logs or other output
                    pass
            except queue.Empty:
                # Check for stderr output
                try:
                    stderr_line = server_info['stderr_queue'].get_nowait()
                    core.logging.log_event(f"MCP Server '{server_name}' stderr: {stderr_line.strip()}", "WARNING")
                except queue.Empty:
                    pass

                # Check if process died
                if server_info['process'].poll() is not None:
                     return {"error": {"message": f"Server '{server_name}' terminated unexpectedly."}}

                time.sleep(0.1)

        return {"error": {"message": f"Timeout waiting for response to request '{request_id}'."}}

    def stop_all_servers(self):
        """Stops all running MCP servers."""
        with self.lock:
            server_names = list(self.servers.keys())
        for name in server_names:
            self.stop_server(name)
        core.logging.log_event("All MCP servers stopped.", "INFO")
    
    def reload_servers(self):
        """
        Hot-reload server configurations from mcp_servers.json.
        Epic 3: Story 3.2 - Enables adding new servers without restart.
        
        Returns:
            dict with 'added', 'removed', and 'unchanged' server names
        """
        old_configs = set(self.server_configs.keys())
        self.server_configs = self._load_server_configs()
        new_configs = set(self.server_configs.keys())
        
        added = new_configs - old_configs
        removed = old_configs - new_configs
        unchanged = old_configs & new_configs
        
        result = {
            "added": list(added),
            "removed": list(removed),
            "unchanged": list(unchanged)
        }
        
        core.logging.log_event(
            f"MCP configs reloaded: {len(added)} added, {len(removed)} removed, {len(unchanged)} unchanged",
            "INFO"
        )
        
        return result
    
    def add_server_config(self, server_name: str, config: dict) -> bool:
        """
        Dynamically add a new server configuration and save to mcp_servers.json.
        Epic 3: Story 3.2 - Autonomous server installation.
        
        Args:
            server_name: Name for the new server
            config: Configuration dict with 'command', 'args', 'requires_env', 'tools'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate config structure
            required_keys = ["command"]
            for key in required_keys:
                if key not in config:
                    core.logging.log_event(
                        f"Invalid server config: missing '{key}'",
                        "ERROR"
                    )
                    return False
            
            # Load current configs
            config_path = "mcp_servers.json"
            current_configs = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    current_configs = json.load(f)
            
            # Add new config
            current_configs[server_name] = config
            
            # Write back
            with open(config_path, 'w') as f:
                json.dump(current_configs, f, indent=4)
            
            # Reload configs
            self.reload_servers()
            
            core.logging.log_event(
                f"Added new MCP server config: {server_name}",
                "INFO"
            )
            
            return True
            
        except Exception as e:
            core.logging.log_event(
                f"Failed to add server config '{server_name}': {e}",
                "ERROR"
            )
            return False

