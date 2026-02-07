
import subprocess
import json
import os
import sys
import re
import time

def get_vram_mb():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(output.strip().split('\n')[0])
    except Exception:
        return 0

def get_context_from_logs():
    log_path = os.path.join(os.getcwd(), "logs", "vllm.log")
    if not os.path.exists(log_path):
        return None
    
    try:
        # Read the last few hundred lines
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-500:] # Check last 500 lines
            
        # Pattern: "Auto-fit max_model_len: .* (\d+) fits"
        # Example: "Auto-fit max_model_len: full model context length 4096 fits in available GPU memory"
        pattern = re.compile(r"Auto-fit max_model_len:.*?(\d+)\s+fits")
        
        for line in reversed(lines):
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error parsing logs: {e}")
        
    return None

def update_config():
    # Try to get from logs first (most accurate)
    # Give vLLM a moment to write logs if this script runs immediately after start
    context_window = get_context_from_logs()
    
    if context_window:
        print(f"Detected context from logs: {context_window}")
    else:
        print("Could not detect context from logs, falling back to VRAM heuristic.")
        vram = get_vram_mb()
        print(f"Detected VRAM: {vram} MB")

        if vram < 10000: # < 10GB
            context_window = 4096
        elif vram < 24000: # < 24GB
            context_window = 8192
        else: # >= 24GB
            context_window = 32768
            
    # Safe max tokens for generation
    max_tokens = 1024

    config = {
        "context_window": context_window,
        "max_tokens": max_tokens
    }

    # Load existing to preserve model_id if possible
    config_path = os.path.join(os.getcwd(), ".vllm_extension_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                existing = json.load(f)
                if "model_id" in existing:
                    config["model_id"] = existing["model_id"]
        except:
            pass

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated vLLM extension config: {config}")

if __name__ == "__main__":
    # Wait a brief moment to ensure logs are flushed if called immediately
    time.sleep(2)
    update_config()
