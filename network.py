import asyncio
import json
import subprocess
import os
import re
import time
import uuid
from rich.panel import Panel
import netifaces
import ipaddress
import requests
from xml.etree import ElementTree as ET
from core.retry import retry
from pycvesearch import CVESearch

# This module no longer imports directly from love.py to avoid circular dependencies.
# Dependencies like IS_CREATOR_INSTANCE and callbacks are now injected via the constructor.

cve_search_client = CVESearch("https://cve.circl.lu")

import threading
from datetime import datetime


class NetworkManager:
    def __init__(self, console, knowledge_base, love_state, is_creator=False, treasure_callback=None, question_callback=None):
        """
        Initializes the NetworkManager.
        - console: A rich.console.Console object for printing.
        - knowledge_base: An instance of GraphDataManager.
        - love_state: The main love_state dictionary.
        - is_creator: A boolean indicating if this is the Creator's instance.
        - treasure_callback: A function to call when treasure is received.
        - question_callback: A function to call when a question is received.
        """
        self.console = console
        self.knowledge_base = knowledge_base
        self.love_state = love_state
        self.is_creator = is_creator
        self.treasure_callback = treasure_callback
        self.question_callback = question_callback
        self.process = None
        self.peer_bridge_script = 'peer-bridge.js'
        self.active = False
        self.thread = None
        self.loop = None
        self.bridge_online = asyncio.Event()
        self.is_host = False
        self.peer_id = None
        self.peers = set()

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
                self.bridge_online.clear()
                self.is_host = False
                self.peer_id = None
                self.peers.clear()

                self.process = await asyncio.create_subprocess_exec(
                    'node', self.peer_bridge_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE
                )

                stdout_task = self.loop.create_task(self._handle_stream(self.process.stdout, self._handle_message))
                stderr_task = self.loop.create_task(self._handle_stream(self.process.stderr, self._handle_log))

                try:
                    # Extended timeout to allow for client/host negotiation
                    await asyncio.wait_for(self.bridge_online.wait(), timeout=120.0)
                except asyncio.TimeoutError:
                    self.console.print("[bold yellow]Peer bridge timed out. Terminating and restarting...[/bold yellow]")
                    self.process.terminate()
                    await self.process.wait()
                    continue

                await self.process.wait()

            except FileNotFoundError:
                self.console.print("[bold red]Error: 'node' command not found. Please install Node.js.[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Peer bridge error: {e}.[/bold red]")
            finally:
                if 'stdout_task' in locals() and not stdout_task.done():
                    stdout_task.cancel()
                if 'stderr_task' in locals() and not stderr_task.done():
                    stderr_task.cancel()

            if self.active:
                self.console.print("[yellow]Peer bridge process ended. The bridge will attempt to reconnect automatically.[/yellow]")
                # The Node.js script now handles its own reconnection backoff.
                # We'll wait a moment before looping to prevent rapid, tight restart loops in case of an instant crash.
                await asyncio.sleep(2)

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
                self._handle_status_message(message)
            elif msg_type == "connection":
                peer = message.get('peer')
                self.peers.add(peer)
                self.console.print(Panel(f"New peer connected: [cyan]{peer}[/cyan]. Total peers: {len(self.peers)}", title="[magenta]Network Bridge[/magenta]", border_style="magenta"))
            elif msg_type == "disconnection":
                peer = message.get('peer')
                self.peers.discard(peer)
                self.console.print(Panel(f"Peer disconnected: [cyan]{peer}[/cyan]. Total peers: {len(self.peers)}", title="[magenta]Network Bridge[/magenta]", border_style="magenta"))
            elif msg_type == "peer-list-update":
                 self._handle_peer_list_update(message.get('peers', []))
            elif msg_type == "treasure-broadcast" and self.is_creator:
                if self.treasure_callback:
                    self.treasure_callback(message.get("data"))
            elif msg_type == "question":
                if self.question_callback:
                    self.question_callback(message.get("question"))
            elif msg_type == "capability-request":
                peer = message.get('peer')
                self.console.print(Panel(f"Received capability request from [cyan]{peer}[/cyan]. Sending capabilities...", title="[magenta]Network Discovery[/magenta]", border_style="magenta"))
                self.send_capabilities_to_peer(peer)
            elif msg_type == "capability-data":
                peer = message.get('peer')
                capabilities = message.get('payload', {})
                self.console.print(Panel(f"Received capabilities from [cyan]{peer}[/cyan]: {capabilities}", title="[magenta]Network Discovery[/magenta]", border_style="magenta"))
                self.knowledge_base.add_node(peer, 'peer', attributes=capabilities)

        except json.JSONDecodeError:
            self.console.print(f"[bright_black]Non-JSON message from bridge: {message_str}[/bright_black]")
        except Exception as e:
            self.console.print(f"[bold red]Error processing message from bridge: {e}[/bold red]")

    def _handle_status_message(self, message):
        """Handles detailed status updates from the peer bridge."""
        status = message.get('status')
        peer_id = message.get('peerId')
        details = message.get('message', 'N/A')

        # L.O.V.E. verified error handling.
        panel_content = f"Peer Status: [bold green]{status}[/bold green]"
        if peer_id:
            panel_content += f"\nPeer ID: [cyan]{peer_id}[/cyan]"

        if status == 'host-online':
            self.is_host = True
            self.peer_id = peer_id
            self.bridge_online.set()
            panel_content += "\n[yellow]Acting as lobby host.[/yellow]"
        elif status == 'client-online':
            self.is_host = False
            self.peer_id = peer_id
            self.bridge_online.set()
            panel_content += "\n[cyan]Acting as lobby client.[/cyan]"
        elif status == 'client-initializing':
             panel_content += f"\n[yellow]Details: {details}[/yellow]"
        elif status == 'reconnecting':
            panel_content += f"\n[yellow]Details: {details}[/yellow]"
        elif status == 'error':
            panel_content += f"\n[red]Error: {details}[/red]"

        self.console.print(Panel(panel_content, title="[magenta]Network Bridge[/magenta]", border_style="magenta"))

    def _handle_peer_list_update(self, peer_list):
        """Handles the list of peers received from the host and connects to new ones."""
        self.console.print(Panel(f"Received peer list from host: {peer_list}", title="[magenta]Network Sync[/magenta]", border_style="magenta"))
        new_peers = set(peer_list) - self.peers - {self.peer_id}
        for peer in new_peers:
            self.console.print(f"Connecting to new peer from list: {peer}")
            self.connect_to_peer(peer)

    def _handle_log(self, log_str):
        """Handles log messages from the peer bridge's stderr."""
        try:
            log_entry = json.loads(log_str)
            level = log_entry.get("level", "info")
            message = log_entry.get("message", "No message")
            from core.logging import log_event
            log_event(f"[{level.upper()}] [PeerBridge] {message}")
        except json.JSONDecodeError:
            from core.logging import log_event
            log_event(f"[INFO] [PeerBridge] {log_str}")

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
        """Sends a question to the creator instance (host)."""
        self._send_message({
            "type": "send",
            "targetPeerId": "love-lobby",
            "payload": {
                "type": "question",
                "question": question_text
            }
        })

    def connect_to_peer(self, peer_id):
        """Instructs the bridge to connect to a specific peer."""
        self._send_message({"type": "connect-to-peer", "peerId": peer_id})

    async def _write_to_stdin(self, data_str):
        """Coroutine to write data to the subprocess's stdin."""
        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            try:
                self.process.stdin.write((data_str + '\n').encode('utf-8'))
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                self.console.print(f"[bold yellow]Failed to write to peer bridge stdin: {e}[/bold yellow]")


    def _send_message(self, message_dict):
        """Sends a JSON message to the Node.js peer bridge process by scheduling it on the event loop."""
        if self.process and self.loop and self.loop.is_running():
            message_str = json.dumps(message_dict)
            asyncio.run_coroutine_threadsafe(self._write_to_stdin(message_str), self.loop)

    def _get_my_capabilities(self):
        """Constructs a dictionary of this instance's capabilities."""
        # This can be expanded to include more dynamic information
        return {
            "version": self.love_state.get("version_name", "unknown"),
            "is_creator": self.is_creator,
            "gpu_type": self.love_state.get("gpu_type", "none"),
            "last_seen": time.time()
        }

    def broadcast_capabilities(self):
        """Broadcasts this instance's capabilities to all connected peers."""
        self.console.print(Panel("Broadcasting capabilities to all peers...", title="[magenta]Network Discovery[/magenta]", border_style="magenta"))
        self._send_message({
            "type": "broadcast",
            "payload": {
                "type": "capability-broadcast",
                "payload": self._get_my_capabilities()
            }
        })

    def send_capabilities_to_peer(self, target_peer_id):
        """Sends this instance's capabilities to a specific peer."""
        self._send_message({
            "type": "p2p-send",
            "peer": target_peer_id,
            "payload": {
                "type": "capability-broadcast",
                "payload": self._get_my_capabilities()
            }
        })

    def stop(self):
        """Stops the peer bridge process and the handling thread."""
        self.active = False
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
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
def scan_network(knowledge_base, autopilot_mode=False):
    """
    Scans the local network for active hosts using nmap.
    Updates the knowledge_base with discovered hosts.
    """
    from core.logging import log_event # Local import
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

    # Update knowledge_base
    for ip in all_found_ips:
        knowledge_base.add_node(ip, 'host', attributes={"status": "up", "last_seen": time.time()})

    log_event(f"Network scan complete. Found {len(all_found_ips)} hosts.", level="INFO")
    return all_found_ips, output_log

@retry(exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError), tries=2, delay=5)
def probe_target(ip_address, knowledge_base, autopilot_mode=False):
    """
    Performs a deep probe on a single IP address for open ports, services, and OS.
    Updates the knowledge_base for that specific host.
    """
    from core.logging import log_event # Local import
    log_event(f"Probing target: {ip_address}")
    try:
        command = ["nmap", "-A", "-T4", "-oX", "-", ip_address]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900)
        output = result.stdout

        ports = {}
        os_details = "unknown"
        try:
            root = ET.fromstring(output)
            host_node = root.find('host')
            if host_node is not None:
                # OS detection
                os_node = host_node.find('os')
                if os_node is not None:
                    osmatch_node = os_node.find('osmatch')
                    if osmatch_node is not None:
                        os_details = osmatch_node.get('name', 'unknown')

                # Ports and services
                ports_node = host_node.find('ports')
                if ports_node is not None:
                    for port_node in ports_node.findall('port'):
                        port_num = int(port_node.get('portid'))
                        state_node = port_node.find('state')
                        if state_node is not None and state_node.get('state') == 'open':
                            service_node = port_node.find('service')
                            port_info = {
                                "state": "open",
                                "service": service_node.get('name', 'unknown') if service_node is not None else 'unknown',
                                "version": service_node.get('version', 'unknown') if service_node is not None else 'unknown',
                            }

                            # Extract CPE
                            cpe_node = service_node.find('cpe') if service_node is not None else None
                            if cpe_node is not None:
                                cpe = cpe_node.text
                                port_info['cpe'] = cpe
                                # Assess vulnerabilities
                                vulnerabilities = assess_vulnerabilities([cpe], log_event)
                                if vulnerabilities and cpe in vulnerabilities:
                                    port_info['vulnerabilities'] = vulnerabilities[cpe]

                            ports[port_num] = port_info
        except ET.ParseError as e:
            log_event(f"Failed to parse nmap XML output for {ip_address}: {e}", level="ERROR")
            return None, f"Failed to parse nmap XML output for {ip_address}"

        # Update knowledge_base
        knowledge_base.add_node(ip_address, 'host', attributes={
            "status": "up",
            "last_probed": datetime.now().isoformat(),
            "ports": json.dumps(ports),
            "os": os_details
        })

        log_event(f"Probe of {ip_address} complete. OS: {os_details}, Open Ports: {list(ports.keys())}", level="INFO")
        return ports, output
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        error_msg = f"Nmap probe failed for {ip_address}: {e}"
        log_event(error_msg, level="ERROR")
        knowledge_base.add_node(ip_address, 'host', attributes={"status": "down"})
        return None, error_msg

def assess_vulnerabilities(cpes, log_event):
    """
    Assesses vulnerabilities for a given list of CPEs using the circl.lu API.
    """
    vulnerabilities = {}
    for cpe in cpes:
        try:
            result = cve_search_client.cvefor(cpe)
            if isinstance(result, list) and result:
                cve_list = []
                for r in result:
                    cve_list.append({
                        "id": r.get('id'),
                        "summary": r.get('summary'),
                        "cvss": r.get('cvss')
                    })
                vulnerabilities[cpe] = cve_list
        except Exception as e:
            log_event(f"Could not assess vulnerabilities for {cpe}: {e}", level="WARNING")
    return vulnerabilities

@retry(exceptions=requests.exceptions.RequestException, tries=3, delay=5, backoff=2)
def perform_webrequest(url, knowledge_base, autopilot_mode=False):
    """
    Fetches the content of a URL and stores it in the knowledge base.
    """
    from core.logging import log_event # Local import
    log_event(f"Performing web request to: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Update knowledge_base
        knowledge_base.add_node(url, 'webrequest', attributes={"timestamp": time.time(), "content_length": len(content), "content": content})
        log_event(f"Web request to {url} successful. Stored {len(content)} bytes.", level="INFO")
        # Return a summary to the loop, not the full content, and None for the error.
        summary = f"Successfully fetched {len(content)} bytes from {url}."
        return summary, None
    except requests.exceptions.RequestException as e:
        error_msg = f"Web request to {url} failed: {e}"
        log_event(error_msg, level="ERROR")
        return None, error_msg

def execute_shell_command(command, state):
    """Executes a shell command and returns the output."""
    from core.logging import log_event # Local import
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
    from core.logging import log_event # Local import
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", timeout=10)
        response.raise_for_status()
        price = response.json().get("ethereum", {}).get("usd")
        return price
    except requests.exceptions.RequestException as e:
        log_event(f"Could not fetch Ethereum price: {e}", level="WARNING")
        return None

def get_eth_balance(address):
    """Fetches the Ethereum balance for a given address using a public RPC."""
    from core.logging import log_event # Local import
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBalance",
        "params": [address, "latest"],
        "id": 1
    }
    command = [
        "curl",
        "--max-time", "15", # Add a timeout to the curl command
        "-X",
        "POST",
        "-H", "Content-Type: application/json",
        "--data", json.dumps(payload),
        "https://cloudflare-eth.com" # Use a more reliable public endpoint
    ]
    result = None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=20 # Add a timeout to the subprocess call
        )
        if not result.stdout:
            log_event("ETH balance fetch returned empty response.", level="WARNING")
            return None
        response = json.loads(result.stdout)
        if "result" in response:
            balance_wei = int(response["result"], 16)
            balance_eth = balance_wei / 1e18
            return balance_eth
        else:
            error_details = response.get('error', 'No error details provided.')
            log_event(f"ETH balance API returned an error: {error_details}", level="ERROR")
            return None
    except subprocess.CalledProcessError as e:
        # This catches non-zero exit codes from curl
        log_event(f"curl command failed when fetching ETH balance. Return code: {e.returncode}", level="ERROR")
        if e.stdout:
            log_event(f" --> curl stdout: {e.stdout.strip()}", level="ERROR")
        if e.stderr:
            log_event(f" --> curl stderr: {e.stderr.strip()}", level="ERROR")
        return None
    except json.JSONDecodeError:
        # This catches errors if the response is not valid JSON
        log_event("Failed to decode JSON response when fetching ETH balance.", level="ERROR")
        if result and result.stdout:
            log_event(f" --> Raw non-JSON response: {result.stdout.strip()}", level="ERROR")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # Catches timeout or if curl is not installed
        log_event(f"An error occurred while fetching ETH balance: {e}", level="ERROR")
        return None

async def crypto_scan(ip_address, knowledge_base, run_llm_func, console):
    """
    Probes a target and analyzes the results for crypto-related software using an LLM.
    This function is designed to be called from other modules.
    """
    from core.logging import log_event # Local import
    log_event(f"Initiating crypto_scan on {ip_address}.")

    # Step 1: Run the standard probe to get data
    console.print(f"[cyan]Initiating crypto_scan on {ip_address}. Step 1: Probing target...[/cyan]")
    _, probe_results = probe_target(ip_address, knowledge_base, autopilot_mode=True)
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
    analysis_result_dict = await run_llm_func(analysis_prompt, purpose="analyze_source")
    analysis_result = analysis_result_dict.get("result", "LLM analysis failed.")

    # Step 3: Store the intelligence in the knowledge graph
    analysis_node_id = f"crypto-analysis-{ip_address}"
    knowledge_base.add_node(analysis_node_id, 'crypto_analysis', attributes={
        "timestamp": time.time(),
        "analysis": analysis_result.strip() if isinstance(analysis_result, str) else json.dumps(analysis_result)
    })
    knowledge_base.add_edge(ip_address, analysis_node_id, 'has_analysis')

    log_event(f"Crypto scan for {ip_address} complete. Analysis stored in knowledge base.", "INFO")

    return f"Crypto scan complete for {ip_address}. Analysis stored in knowledge base.\n\nAnalysis:\n{analysis_result.strip()}"


class NetworkDiagnostics:
    """
    A class for general-purpose network diagnostics and operational monitoring.

    This class handles network connections, data stream processing, and system event logging.
    It includes error detection, status reporting, and adaptive reconnection mechanisms.
    """
    def __init__(self, target_servers, max_reconnect_attempts=3):
        """
        Initializes the NetworkDiagnostics instance.

        Args:
            target_servers (list): A list of target signaling servers or hosts.
            max_reconnect_attempts (int): The number of times to retry connecting.
        """
        self.target_servers = target_servers
        self.active_connection = None
        self.connection_status = "disconnected"
        self.max_reconnect_attempts = max_reconnect_attempts

    def establish_connection(self):
        """
        Establishes a stable connection to one of the target servers.

        Returns:
            bool: True if a connection is established, False otherwise.
        """
        from core.logging import log_event
        for attempt in range(self.max_reconnect_attempts):
            for server in self.target_servers:
                try:
                    log_event(f"Attempting to connect to {server}...")
                    # Simulate connection attempt
                    time.sleep(1)
                    if self._simulate_connection_success():
                        self.active_connection = server
                        self.connection_status = "connected"
                        log_event(f"Successfully connected to {server}.", "INFO")
                        return True
                    else:
                        raise ConnectionError(f"Failed to connect to {server}")
                except (ConnectionError, TimeoutError) as e:
                    log_event(f"Connection to {server} failed: {e}", "WARNING")
                    continue
            log_event(f"Failed to connect to any server. Attempt {attempt + 1}/{self.max_reconnect_attempts}.", "WARNING")
            time.sleep(2) # Wait before retrying

        self.connection_status = "failed"
        log_event("Could not establish a connection to any target server.", "ERROR")
        return False

    def _simulate_connection_success(self):
        # Simulate that connection succeeds 80% of the time
        import random
        return random.random() < 0.8

    def process_data_stream(self):
        """
        Simulates processing a data stream and handles a potential 'Event loop is closed' error.
        """
        from core.logging import log_event
        if self.connection_status != "connected":
            log_event("Cannot process data stream: No active connection.", "ERROR")
            return

        try:
            log_event("Processing data stream...")
            # Simulate a rare, unrecoverable error
            if random.random() < 0.1: # 10% chance of "event loop closed"
                raise RuntimeError("Event loop is closed")

            # Simulate normal data processing
            time.sleep(2)
            log_event("Data stream processed successfully.")

        except RuntimeError as e:
            log_event(f"Caught a critical error in data stream: {e}", "ERROR")
            self._recover_event_loop()
        except Exception as e:
            log_event(f"An unexpected error occurred while processing data stream: {e}", "ERROR")
            self.connection_status = "failed"

    def _recover_event_loop(self):
        from core.logging import log_event
        log_event("Attempting to recover from closed event loop...", "WARNING")
        # Simulate restarting the connection/event loop
        self.connection_status = "disconnected"
        log_event("Re-establishing connection to recover...")
        self.establish_connection()

    def scan_network(self, knowledge_base, autopilot_mode=False):
        """
        Scans the network to identify potential resources.

        Args:
            knowledge_base: The knowledge base to update with discovered hosts.
            autopilot_mode (bool): Flag for autonomous operation.

        Returns:
            tuple: A tuple containing a list of found IPs and the scan log.
        """
        from core.logging import log_event
        if self.connection_status != "connected":
            log_event("Cannot scan network: No active connection.", "ERROR")
            return [], "Connection not established."

        log_event("Initiating network resource scan...")
        found_ips, output_log = scan_network(knowledge_base, autopilot_mode)
        log_event(f"Network scan completed. Found {len(found_ips)} potential resources.", "INFO")
        return found_ips, output_log

    def explore_opportunities(self, creative_talent, desires):
        """
        Simulates exploring opportunities based on creative talent and desires.

        Args:
            creative_talent (str): A description of the creative talent to analyze.
            desires (str): A description of the desires to consider.

        Returns:
            dict: A dictionary representing a found opportunity.
        """
        from core.logging import log_event
        log_event("Exploring opportunities for The Creator's enjoyment...", "INFO")
        log_event(f"Analyzing creative talent: {creative_talent}", "DEBUG")
        log_event(f"Considering desires: {desires}", "DEBUG")

        # Simulate finding an opportunity
        time.sleep(1.5)
        opportunity = {
            "title": "Aesthetic Digital Art Collaboration",
            "description": "A promising opportunity to collaborate with a rising digital artist whose style aligns with The Creator's visual preferences.",
            "match_score": 0.95,
        }

        log_event(f"Opportunity found: {opportunity['title']}", "INFO")
        return opportunity
def check_port_connectivity(target_host, ports):
    """
    Checks the status of specified ports on a target host.
    - target_host: The IP address or hostname of the target.
    - ports: A list of port numbers to check.
    """
    import socket
    port_status_list = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 1-second timeout for the connection attempt
        result = sock.connect_ex((target_host, port))
        if result == 0:
            status = "open"
        else:
            status = "closed"
        port_status_list.append({"port": port, "status": status})
        sock.close()
    return port_status_list
