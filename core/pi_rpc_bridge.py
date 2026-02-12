import asyncio
import os
import json
import logging
import subprocess
import uuid
import signal
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
        self.callbacks: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self.running = False
    
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
    
    def _write_extension_config(self):
        """Write config file for the vLLM extension with current model info."""
        import requests
        
        model_id = self._get_vllm_model()
        context_window = 4096  # Default
        max_tokens = 1024  # Default (conservative)
        
        # Try to get actual context window from vLLM
        try:
            response = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
            if response.ok:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    model_info = data["data"][0]
                    # vLLM returns max_model_len
                    if "max_model_len" in model_info:
                        context_window = model_info["max_model_len"]
                        max_tokens = max(256, context_window // 4)  # 25% of context
                    # Use actual model ID from vLLM
                    model_id = model_info.get("id", model_id)
                    logger.info(f"Got vLLM model info: {model_id}, context={context_window}")
        except Exception as e:
            logger.warning(f"Failed to query vLLM for model info: {e}")
        
        # Write config for JS extension
        config_path = os.path.join(self.agent_dir, ".vllm_extension_config.json")
        config = {
            "model_id": model_id,
            "context_window": context_window,
            "max_tokens": max_tokens
        }
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Wrote extension config: {config}")
        except Exception as e:
            logger.warning(f"Failed to write extension config: {e}")

    def set_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]], callback_id: str = "default"):
        """Set or update a named callback."""
        self.callbacks[callback_id] = callback

    def remove_callback(self, callback_id: str):
        """Remove a named callback."""
        if callback_id in self.callbacks:
            del self.callbacks[callback_id]

    async def start(self):
        """Start the Pi Agent in RPC mode."""
        # If already running, just return to keep it warm
        if self.running:
            return
        
        # Write extension config with current vLLM model info
        try:
            self._write_extension_config()
        except Exception as e:
            logger.error(f"Failed to write extension config: {e}")

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

            # Convert config path to WSL format
            config_win_path = os.path.join(self.agent_dir, ".vllm_extension_config.json")
            wsl_config_path = to_wsl_path(config_win_path)

            runner_content = f"""#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
export VLLM_EXTENSION_CONFIG_PATH="{wsl_config_path}"
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
                config_path = os.path.join(self.agent_dir, ".vllm_extension_config.json")
                
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
export VLLM_EXTENSION_CONFIG_PATH="{config_path}"
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
            # Set up environment with config path for the extension
            env = os.environ.copy()
            config_path = os.path.join(self.agent_dir, ".vllm_extension_config.json")
            env["VLLM_EXTENSION_CONFIG_PATH"] = config_path
            
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=popen_cwd,
                env=env,
                text=True,
                bufsize=1,
                start_new_session=True
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
        if not self.process:
            self.running = False
            return
            
        start_stop = asyncio.get_event_loop().time()
        logger.info(f"Stopping Pi Agent (PID {self.process.pid})...")
        self.running = False
        
        try:
            # Send abort command first nicely if stdin is open
            if self.process.stdin:
                try:
                    logger.info("Sending 'abort' command to Pi Agent...")
                    self.process.stdin.write(json.dumps({"type": "abort"}) + "\n")
                    self.process.stdin.flush()
                except:
                    pass
            
            await asyncio.sleep(0.1)
            
            # Close pipes to unblock any threads waiting on readline
            logger.info("Closing Pi Agent pipes...")
            for pipe in [self.process.stdin, self.process.stdout, self.process.stderr]:
                if pipe:
                    try: pipe.close()
                    except: pass
            
            # Try SIGTERM to the process group
            logger.info(f"Sending SIGTERM to Pi Agent process group {self.process.pid}...")
            try:
                import os
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                logger.debug(f"SIGTERM process group failed: {e}")
                try:
                    self.process.send_signal(signal.SIGTERM)
                except:
                    self.process.terminate()
            
            try:
                # Wait in thread since wait() is blocking
                logger.info("Waiting for Pi Agent process to exit...")
                await asyncio.to_thread(self.process.wait, timeout=2)
            except:
                # Force kill if still alive
                logger.warning("Pi Agent did not exit gracefully, sending SIGKILL to process group...")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    try: self.process.kill()
                    except: pass
        except Exception as e:
            logger.debug(f"Error during bridge stop: {e}")
        finally:
            self.process = None
            self.running = False
            elapsed = (asyncio.get_event_loop().time() - start_stop)
            logger.info(f"Pi Agent stopped in {elapsed:.2f}s")

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
                # Emit to all registered callbacks
                if self.callbacks:
                    for cid, cb in list(self.callbacks.items()):
                        try:
                            # Use create_task to avoid blocking the read loop
                            asyncio.create_task(cb(data))
                        except Exception as e:
                            logger.error(f"Error in callback {cid}: {e}")
            except json.JSONDecodeError:
                # Non-JSON output (e.g., extension init messages, progress logs)
                logger.info(f"Pi Agent: {line}")
            except Exception as e:
                logger.error(f"Error handling agent output: {e}")
        
        if self.process:
            logger.debug(f"Pi Agent process exited with code: {self.process.poll()}")

    async def _read_stderr(self):
        """Read stderr for logs."""
        if not self.process or not self.process.stderr:
            return
            
        while self.running:
            line = await asyncio.to_thread(self.process.stderr.readline)
            if line:
                logger.debug(f"Pi Log: {line.strip()}")

# Singleton
_bridge: Optional[PiRPCBridge] = None

def get_pi_bridge() -> PiRPCBridge:
    global _bridge
    if not _bridge:
        # Base dir is project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _bridge = PiRPCBridge(base_dir)
    return _bridge
