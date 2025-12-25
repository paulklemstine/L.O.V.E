import os
import signal
import psutil
import core.logging

def restart_vllm_service(deep_agent_instance=None):
    """
    Restarts the vLLM service.
    If deep_agent_instance is provided, it can be used to notify the agent.
    """
    core.logging.log_event("Attempting to restart vLLM service...", "INFO")
    
    # 1. Kill existing vllm process
    killed = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and any('vllm' in cmd for cmd in proc.info['cmdline']):
                core.logging.log_event(f"Killing vLLM process (PID: {proc.info['pid']})...", "INFO")
                proc.kill()
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    if not killed:
        core.logging.log_event("No running vLLM process found to kill.", "WARNING")

    # 2. Restart the service (assuming it's managed by a script or creating a new subprocess)
    # Ideally, we should use a service manager or the original startup script.
    # For now, we will just start the launch command.
    try:
        import subprocess
        # Assuming run_vllm.sh is in the root or scripts dir
        # We need to find the correct path
        script_path = os.path.join(os.getcwd(), "scripts", "run_vllm.sh") # Adjust as needed
        if os.path.exists(script_path):
             subprocess.Popen(["bash", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             return "vLLM service restart initiated."
        else:
            return f"vLLM restart script not found at {script_path}"
    except Exception as e:
        return f"Failed to restart vLLM: {e}"
