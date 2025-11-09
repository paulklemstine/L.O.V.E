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
from core.knowledge_extractor import KnowledgeExtractor

# This module no longer imports directly from love.py to avoid circular dependencies.
# Dependencies like IS_CREATOR_INSTANCE and callbacks are now injected via the constructor.

cve_search_client = CVESearch("https://cve.circl.lu")

import threading
from datetime import datetime




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

        # Enrich knowledge base with parsed data
        extractor = KnowledgeExtractor(knowledge_base)
        extractor.parse_probe_data(ip_address, ports)

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
        knowledge_base.add_node(url, 'webrequest', attributes={"timestamp": time.time(), "content_length": len(content)})
        log_event(f"Web request to {url} successful. Stored {len(content)} bytes.", level="INFO")

        # Enrich knowledge base with parsed data
        extractor = KnowledgeExtractor(knowledge_base)
        extractor.parse_web_content(url, content)

        # Return a summary to the loop, not the full content, and None for the error.
        summary = f"Successfully fetched {len(content)} bytes from {url}."
        return summary, None
    except requests.exceptions.RequestException as e:
        error_msg = f"Web request to {url} failed: {e}"
        log_event(error_msg, level="ERROR")
        return None, error_msg

def execute_shell_command(command, state):
    """
    Executes a shell command and returns the output.
    Intercepts `echo '...' > file.py` commands to bypass shell length limits.
    """
    from core.logging import log_event # Local import
    log_event(f"Executing shell command: {command}")

    # --- Intercept echo > file.py pattern ---
    # This regex captures the content inside single quotes and the filename.
    # It's designed to be robust against variations in spacing.
    match = re.match(r"^\s*echo\s+'(.*)'\s*>\s*([\w\/\.\-]+.py)\s*$", command, re.DOTALL)

    if match:
        content, filename = match.groups()
        log_event(f"Intercepted echo command. Writing {len(content)} chars to '{filename}' directly.", "INFO")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File '{filename}' created successfully.", "", 0
        except (IOError, OSError) as e:
            log_event(f"File write error during echo interception: {e}", level="ERROR")
            return "", str(e), 1

    # --- Standard Command Execution ---
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

    endpoints = [
        "https://cloudflare-eth.com",
        "https://rpc.ankr.com/eth",
        "https://eth.llamarpc.com",
        "https://eth.api.onfinality.io/public",
        "https://ethereum.public.blockpi.network/v1/rpc/public",
        "https://rpc.flashbots.net"
    ]

    for endpoint in endpoints:
        command = [
            "curl",
            "--max-time", "15",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "--data", json.dumps(payload),
            endpoint
        ]
        result = None
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=20
            )
            if not result.stdout:
                log_event(f"ETH balance fetch from {endpoint} returned empty response.", level="WARNING")
                continue
            response = json.loads(result.stdout)
            if "result" in response and response["result"] is not None:
                balance_wei = int(response["result"], 16)
                balance_eth = balance_wei / 1e18
                return balance_eth
            else:
                error_details = response.get('error', 'No error details provided.')
                log_event(f"ETH balance API at {endpoint} returned an error: {error_details}", level="ERROR")
                continue
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            log_event(f"Failed to fetch ETH balance from {endpoint}: {e}", level="WARNING")
            continue

    log_event("All ETH balance endpoints failed.", level="ERROR")
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
    analysis_result_dict = await run_llm_func(analysis_prompt, purpose="analyze_source", force_model=None)
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
