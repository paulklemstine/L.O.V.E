import subprocess
import os
import sys
import platform
import shutil
import logging
import netifaces


def get_network_interfaces(autopilot_mode=False):
    """
    Retrieves detailed information about all network interfaces.
    Returns a tuple of (details_dict, error_string).
    """
    interfaces = {}
    try:
        for iface in netifaces.interfaces():
            details = netifaces.ifaddresses(iface)
            interfaces[iface] = {
                "mac": details.get(netifaces.AF_LINK, [{}])[0].get('addr', 'N/A'),
                "ipv4": details.get(netifaces.AF_INET, [{}])[0],
                "ipv6": details.get(netifaces.AF_INET6, [{}])[0],
            }
        # For autopilot, return a simple string summary
        if autopilot_mode:
            summary_lines = []
            for iface, data in interfaces.items():
                if data['ipv4'].get('addr'):
                    summary_lines.append(f"{iface}: {data['ipv4']['addr']}")
            return interfaces, ", ".join(summary_lines)
        return interfaces, None
    except Exception as e:
        logging.error(f"Could not get network interface details: {e}")
        return None, f"Error getting network interface details: {e}"


def get_git_repo_info():
    """Retrieves the GitHub repository owner and name from any available remote URL."""
    try:
        # Get all remotes
        remotes_result = subprocess.run(
            ["git", "remote"],
            capture_output=True,
            text=True,
            check=True
        )
        remotes = remotes_result.stdout.strip().splitlines()

        if not remotes:
            return None, None

        # Try each remote until we find a valid GitHub URL
        for remote_name in remotes:
            url_result = subprocess.run(
                ["git", "config", "--get", f"remote.{remote_name}.url"],
                capture_output=True,
                text=True,
                check=True
            )
            url = url_result.stdout.strip()

            # Extract owner and repo name
            if "github.com" in url:
                if url.startswith("git@"):
                    # SSH format: git@github.com:owner/repo.git
                    parts = url.split(":")[1].split("/")
                    owner = parts[0]
                    repo = parts[1].replace(".git", "")
                    return owner, repo
                elif url.startswith("https://"):
                    # HTTPS format: https://github.com/owner/repo.git
                    parts = url.split("/")
                    owner = parts[-2]
                    repo = parts[-1].replace(".git", "")
                    return owner, repo

        return None, None  # No GitHub URL found in any remote

    except (subprocess.CalledProcessError, IndexError):
        return None, None


def list_directory(path="."):
    """
    Lists the contents of a directory.
    Returns a tuple of (list_of_contents, error_string).
    """
    if not os.path.isdir(path):
        return None, f"Error: Directory not found at '{path}'"
    try:
        # Use a more informative listing format, similar to 'ls -la'
        result = subprocess.run(
            ["ls", "-la", path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error listing directory '{path}': {e.stderr}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def get_file_content(filepath):
    """
    Reads the content of a file.
    Returns a tuple of (file_content, error_string).
    """
    if not os.path.isfile(filepath):
        return None, f"Error: File not found at '{filepath}'"
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, None
    except Exception as e:
        return None, f"Error reading file '{filepath}': {e}"


def get_process_list():
    """
    Gets a list of running processes using 'ps'.
    Returns a tuple of (process_list_string, error_string).
    """
    try:
        # Use 'ps aux' for a detailed, cross-user process list
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error getting process list: {e.stderr}"
    except FileNotFoundError:
        return None, "Error: 'ps' command not found. Unable to list processes."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def parse_ps_output(ps_output):
    """
    Parses the output of 'ps aux' into a list of dictionaries.
    """
    processes = []
    lines = ps_output.strip().split('\n')
    header = [h.lower() for h in lines[0].split()]
    # Expected header columns: USER, PID, %CPU, %MEM, VSZ, RSS, TTY, STAT, START, TIME, COMMAND
    for line in lines[1:]:
        parts = line.split(None, len(header) - 1)
        if len(parts) == len(header):
            process_info = dict(zip(header, parts))
            processes.append(process_info)
    return processes


def get_cpu_usage():
    """
    Retrieves system-wide CPU usage percentage.
    Returns a dictionary with cpu usage details or an error string.
    """
    try:
        # Use 'top' to get a snapshot of CPU usage. -b is for batch mode, -n1 for one iteration.
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
            check=True
        )
        cpu_line = [line for line in result.stdout.split('\n') if line.startswith('%Cpu(s)')][0]
        # Example: %Cpu(s):  0.3 us,  0.1 sy,  0.0 ni, 99.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
        idle_search = __import__('re').search(r'(\d+\.\d+)\s+id', cpu_line)
        if idle_search:
            idle_percent = float(idle_search.group(1))
            usage_percent = 100.0 - idle_percent
            return {"cpu_usage_percent": round(usage_percent, 2)}, None
        return None, "Could not parse CPU idle percentage from 'top' command."
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError) as e:
        return None, f"Error getting CPU usage: {e}"


def get_memory_usage():
    """
    Retrieves system memory (RAM) and swap usage.
    Returns a dictionary with memory details in MB or an error string.
    """
    try:
        # Use 'free -m' to get memory usage in megabytes.
        result = subprocess.run(
            ["free", "-m"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        mem_line = [line for line in lines if line.startswith('Mem:')][0]
        swap_line = [line for line in lines if line.startswith('Swap:')][0]

        mem_parts = mem_line.split()
        swap_parts = swap_line.split()

        memory_data = {
            "ram_total_mb": int(mem_parts[1]),
            "ram_used_mb": int(mem_parts[2]),
            "ram_free_mb": int(mem_parts[3]),
            "ram_used_percent": round((int(mem_parts[2]) / int(mem_parts[1])) * 100, 2)
        }
        # Swap may not exist, so total can be 0.
        if int(swap_parts[1]) > 0:
            swap_used_percent = round((int(swap_parts[2]) / int(swap_parts[1])) * 100, 2)
        else:
            swap_used_percent = 0

        swap_data = {
            "swap_total_mb": int(swap_parts[1]),
            "swap_used_mb": int(swap_parts[2]),
            "swap_free_mb": int(swap_parts[3]),
            "swap_used_percent": swap_used_percent
        }

        return {"memory": memory_data, "swap": swap_data}, None
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError) as e:
        return None, f"Error getting memory usage: {e}"


def get_disk_usage(path="/"):
    """
    Retrieves disk usage for a given path.
    Returns a dictionary with disk usage details or an error string.
    """
    try:
        # Use 'df -h' for human-readable disk usage statistics.
        result = subprocess.run(
            ["df", "-h", path],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None, "Invalid output from 'df' command."

        # e.g., /dev/vda1        74G   20G   55G  27% /
        parts = lines[1].split()
        if len(parts) < 5:
             return None, f"Could not parse 'df' output: {lines[1]}"

        disk_data = {
            "filesystem": parts[0],
            "total_space": parts[1],
            "used_space": parts[2],
            "available_space": parts[3],
            "used_percent": parts[4].replace('%', '')
        }
        return disk_data, None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return None, f"Error getting disk usage: {e}"