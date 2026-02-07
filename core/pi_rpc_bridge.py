
import asyncio
import os
import json
import logging
import subprocess
import uuid
from typing import Optional, Callable, Dict, Any, Awaitable

logger = logging.getLogger("PiRPCBridge")

class PiRPCBridge:
    """
    Bridge to communicate with the Pi Agent via RPC/JSON-RPC.
    """
    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    
    def __init__(self, agent_dir: str):
        self.agent_dir = agent_dir
        self.process: Optional[subprocess.Popen] = None
        self.event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self.running = False
        self.loop = asyncio.get_event_loop()
    
    def _get_vllm_model(self) -> str:
        """Get the model name from .vllm_config, or use default."""
        config_path = os.path.join(self.agent_dir, ".vllm_config")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    model_name = config.get("model_name", self.DEFAULT_MODEL)
                    logger.info(f"Using model from .vllm_config: {model_name}")
                    return model_name
        except Exception as e:
            logger.warning(f"Failed to read .vllm_config: {e}")
        return self.DEFAULT_MODEL

    def set_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        self.event_callback = callback

    async def start(self):
        """Start the Pi Agent in RPC mode."""
        if self.running:
            return

        # Check if we are potentially dealing with a WSL path on Windows
        is_wsl_share = self.agent_dir.startswith(r"\\wsl")
        
        # Path processing helper
        def to_wsl_path(win_path):
            # Convert \\wsl.localhost\Ubuntu\home\user... to /home/user...
            # Simplistic regex or split
            parts = win_path.split('\\')
            # \\wsl.localhost\Distro\path...
            if len(parts) >= 4:
                return "/" + "/".join(parts[4:])
            return win_path

        # Determine CLI path (host OS path)
        pi_cli_dist = os.path.join(self.agent_dir, "external", "pi-agent", "packages", "coding-agent", "dist", "cli.js")
        pi_cli_src = os.path.join(self.agent_dir, "external", "pi-agent", "packages", "coding-agent", "src", "cli.ts")
        
        cwd = os.path.join(self.agent_dir, "external", "pi-agent")
        
        if is_wsl_share:
            # --- WINDOWS ACCESSING WSL ---
            # We are on Windows accessing WSL
            # Check file existence using os.path (Windows)
            use_src = not os.path.exists(pi_cli_dist) and os.path.exists(pi_cli_src)
            
            # Paths for the command must be WSL paths
            wsl_cwd = to_wsl_path(cwd)
            wsl_extension_path = to_wsl_path(os.path.join(self.agent_dir, "external", "vllm-extension"))
            wsl_target_cli = to_wsl_path(pi_cli_src if use_src else pi_cli_dist)
            
            distro = self.agent_dir.split('\\')[3] # Extract distro name
            
            # Construct the inner command string to run inside bash
            # We used to try inline sourcing, but it's brittle with quoting.
            # Instead, we will rely on a resolved node path.
            
            # Helper to resolve node path if not cached
            wsl_node_path = "node" # Default fallback
            
            try:
                # Create a temporary script to find node
                # We write this to the agent_dir which is accessible to both
                script_content = """#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
which node
"""
                script_path = os.path.join(self.agent_dir, "wsl_find_node.sh")
                wsl_script_path = wsl_cwd + "/wsl_find_node.sh" # Approximation
                
                # Check formatting. wsl_cwd from to_wsl_path might act as a base
                # wsl_cwd is like /home/user/L.O.V.E/external/pi-agent
                # agent_dir is the root L.O.V.E
                
                # We need to write to the file system.
                with open(os.path.join(self.agent_dir, "wsl_find_node.sh"), "w", newline='\n') as f:
                    f.write(script_content)
                
                # Execute it
                # We need to determine the path to the script inside WSL.
                # agent_dir is \\wsl.localhost\Ubuntu\home\raver1975\L.O.V.E
                # wsl_script_path should be /home/raver1975/L.O.V.E/wsl_find_node.sh
                
                # Re-use to_wsl_path logic
                wsl_script_absolute = to_wsl_path(os.path.join(self.agent_dir, "wsl_find_node.sh"))
                
                # Run bash to execute it
                # wsl -d distro -- bash wsl_script_absolute
                probe_cmd = ["wsl", "-d", distro, "--", "bash", wsl_script_absolute]
                
                # logger.info(f"Probing for node: {' '.join(probe_cmd)}")
                
                probe_proc = subprocess.run(
                    probe_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=os.environ.get("TEMP", "C:\\")
                )
                
                if probe_proc.returncode == 0:
                    found_path = probe_proc.stdout.strip().split('\n')[-1] # Take last line
                    if found_path and not "not found" in found_path:
                        wsl_node_path = found_path
                        logger.info(f"Resolved WSL Node path: {wsl_node_path}")
            except Exception as e:
                logger.warning(f"Failed to resolve specific node path in WSL: {e}")

            inner_cmd_parts = []
            if use_src:
                # If using src, we ideally want the resolved node to run tsx?
                # npx usually finds node in path.
                # If we use robust resolving, we might just export PATH in the wrapper? 
                # Let's try just using the raw node path for 'node' calls, 
                # and for npx we might still need to source.
                
                # But wait, if we found 'node', we can probably assume its dir is valid.
                pass
            
            # Actually, if we use the absolute path to `node`, we don't need `nvm use`!
            # But we might need other env vars.
            
            # Let's go back to the single-command strategy but using a generated script file
            # instead of inline quoting hell.
            
            # Create the runner script
            runner_script_name = "wsl_run_pi.sh"
            
            wsl_target_cli_quoted = f'"{wsl_target_cli}"'
            
            if use_src:
                # For src we need npx. We can assume npx is in same dir as node?
                # Or just source nvm again in the runner script (it works fine in a file)
                cmd_str = f'npx -y tsx {wsl_target_cli_quoted} --mode rpc --extension "{wsl_extension_path}" --provider vllm --model {self._get_vllm_model()}'
            else:
                cmd_str = f'node {wsl_target_cli_quoted} --mode rpc --extension "{wsl_extension_path}" --provider vllm --model {self._get_vllm_model()}'

            runner_content = f"""#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
{cmd_str}
"""
            with open(os.path.join(self.agent_dir, runner_script_name), "w", newline='\n') as f:
                f.write(runner_content)
                
            wsl_runner_path = to_wsl_path(os.path.join(self.agent_dir, runner_script_name))
            
            # Final command: execute the runner script
            # Final command: execute the runner script
            command = ["wsl", "-d", distro, "--cd", wsl_cwd, "--", "bash", wsl_runner_path]
            
            logger.info(f"Starting Pi Agent (WSL setup): {' '.join(command)}")
            
            # Set Popen CWD to a safe local dir to avoid UNC errors
            popen_cwd = os.environ.get("TEMP", "C:\\")
            
            # Set Popen CWD to a safe local dir to avoid UNC errors
            popen_cwd = os.environ.get("TEMP", "C:\\")

        else:
            # --- NATIVE (LINUX OR WINDOWS LOCAL) ---
            use_src = not os.path.exists(pi_cli_dist) and os.path.exists(pi_cli_src)
            extension_path = os.path.join(self.agent_dir, "external", "vllm-extension")
            
            # Determine target CLI path
            target_cli = pi_cli_src if use_src else pi_cli_dist
            
            # On Windows, we might find node in path no problem. 
            # On Linux (Native), we often miss NVM.
            # We will use the same script strategy if on POSIX to be safe.
            
            if os.name == "nt":
                # Windows Local Execution
                cmd_prefix = []
                if use_src:
                    logger.info("Falling back to source with npx tsx...")
                    npx = "npx.cmd"
                    cmd_prefix = [npx, "-y", "tsx"]
                else:
                    cmd_prefix = ["node"]
                    
                command = cmd_prefix + [
                    target_cli,
                    "--mode", "rpc",
                    "--extension", extension_path,
                    "--provider", "vllm", 
                    "--model", self._get_vllm_model(),
                ]
                
                logger.info(f"Starting Pi Agent (Native Windows): {' '.join(command)}")
                popen_cwd = cwd
                
            else:
                # LINUX Native (likely WSL direct or Linux box)
                # Use wrapper script to source NVM
                
                runner_script_name = "run_pi_native.sh"
                
                # construct inner command
                if use_src:
                    # npx tsx ...
                    # We assume npx is in path after nvm use
                    cmd_str = f'npx -y tsx "{target_cli}" --mode rpc --extension "{extension_path}" --provider vllm --model {self._get_vllm_model()}'
                else:
                    cmd_str = f'node "{target_cli}" --mode rpc --extension "{extension_path}" --provider vllm --model {self._get_vllm_model()}'

                runner_content = f"""#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
{cmd_str}
"""
                output_script = os.path.join(self.agent_dir, runner_script_name)
                # Write script
                with open(output_script, "w", newline='\n') as f:
                    f.write(runner_content)
                
                # Ensure executable
                st = os.stat(output_script)
                os.chmod(output_script, st.st_mode | 0o111)
                
                # Command is just the script, wrapped in stdbuf
                command = ["stdbuf", "-oL", "-eL", "bash", output_script]
                
                logger.info(f"Starting Pi Agent (Native Linux Helper): {' '.join(command)}")
                popen_cwd = cwd

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=popen_cwd,
                text=True,
                bufsize=1 
            )
            self.running = True
            
            # Start reading output in background
            asyncio.create_task(self._read_stdout())
            asyncio.create_task(self._read_stderr())
            
            logger.info("Pi Agent process started.")
            
        except Exception as e:
            logger.error(f"Failed to start Pi Agent: {e}")
            self.running = False

    async def stop(self):
        """Stop the agent."""
        if self.process:
            logger.info("Stopping Pi Agent...")
            if self.running:
                # Send abort command first nicely
                await self.send_command_json({"type": "abort"})
                await asyncio.sleep(0.5)
            
            self.process.terminate()
            try:
                # Wait in tread since wait() is blocking
                await asyncio.to_thread(self.process.wait, timeout=5)
            except:
                self.process.kill()
        self.running = False

    async def send_prompt(self, message: str):
        """Send a prompt to the agent."""
        cmd = {
            "id": str(uuid.uuid4()),
            "type": "prompt",
            "message": message
        }
        await self.send_command_json(cmd)

    async def send_command_json(self, cmd: Dict[str, Any]):
        """Send a raw JSON command."""
        if not self.process:
            logger.warning("Agent not running, cannot send command.")
            return

        json_line = json.dumps(cmd) + "\n"
        
        # Write to stdin in thread to avoid blocking loop
        def _write():
            try:
                self.process.stdin.write(json_line)
                self.process.stdin.flush()
            except Exception as e:
                logger.error(f"Failed to write to agent stdin: {e}")
        
        await asyncio.to_thread(_write)

    async def _read_stdout(self):
        """Read stdout and emit events."""
        if not self.process or not self.process.stdout:
            return
            
        logger.info("Started reading agent stdout.")
        while self.running:
            line = await asyncio.to_thread(self.process.stdout.readline)
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                # Try parsing as JSON event
                data = json.loads(line)
                if self.event_callback:
                    await self.event_callback(data)
            except json.JSONDecodeError:
                # If not JSON, just log it (maybe part of a stream we missed?)
                logger.info(f"Pi Output (Raw): {line}")
            except Exception as e:
                logger.error(f"Error handling agent output: {e}")
        
        logger.info(f"Finished reading agent stdout. Process return code: {self.process.poll()}")

    async def _read_stderr(self):
        """Read stderr for logs."""
        if not self.process or not self.process.stderr:
            return
            
        while self.running:
            line = await asyncio.to_thread(self.process.stderr.readline)
            if line:
                logger.warning(f"Pi Log: {line.strip()}")

# Singleton
_bridge: Optional[PiRPCBridge] = None

def get_pi_bridge() -> PiRPCBridge:
    global _bridge
    if not _bridge:
        # Base dir is project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _bridge = PiRPCBridge(base_dir)
    return _bridge
