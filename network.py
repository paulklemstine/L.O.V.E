import subprocess
import json
import logging
import re
import ipaddress
import socket
import shutil
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import netifaces
from rich.console import Console
from rich.panel import Panel

from bbs import run_hypnotic_progress
from ipfs import pin_to_ipfs, verify_ipfs_pin

class NetworkManager(Thread):
    """
    Manages the Node.js peer-bridge.js script as a subprocess, handling
    the JSON-based communication between Python and the Node.js process.
    """
    def __init__(self, console=None):
        super().__init__()
        self.daemon = True
        self.console = console if console else Console()
        self.bridge_process = None
        self.peer_id = None
        self.online = False

    def run(self):
        """Starts the node bridge and threads to read its stdout/stderr."""
        logging.info("Starting Node.js peer bridge...")
        try:
            self.bridge_process = subprocess.Popen(
                ['node', 'peer_bridge.js'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1, # Line-buffered
                encoding='utf-8'
            )
            Thread(target=self._read_stdout, daemon=True).start()
            Thread(target=self._read_stderr, daemon=True).start()
        except FileNotFoundError:
            logging.error("Node.js is not installed or not in PATH. P2P functionality disabled.")
            self.console.print("[bold red]Error: 'node' command not found. P2P features will be disabled.[/bold red]")
            self.online = False
        except Exception as e:
            logging.critical(f"Failed to start peer_bridge.js: {e}")
            self.console.print(f"[bold red]Critical error starting Node.js bridge: {e}[/bold red]")
            self.online = False

    def _read_stdout(self):
        """Reads and processes JSON messages from the Node.js bridge's stdout."""
        for line in iter(self.bridge_process.stdout.readline, ''):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                msg_type = message.get('type')

                if msg_type == 'status':
                    self._handle_status(message)
                elif msg_type == 'pin-request':
                    self._handle_pin_request(message)
                else:
                    logging.warning(f"Received unknown message type from bridge: {msg_type}")

            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON from node bridge: {line.strip()}")
            except Exception as e:
                logging.error(f"Error processing message from bridge: {e}\nMessage: {line.strip()}")

    def _read_stderr(self):
        """Reads and processes structured JSON logs from the Node.js bridge's stderr."""
        for line in iter(self.bridge_process.stderr.readline, ''):
            if not line.strip():
                continue
            try:
                log_entry = json.loads(line)
                log_level = log_entry.get('level', 'info').upper()
                log_message = log_entry.get('message', 'No message content.')

                # Log to file
                logging.log(getattr(logging, log_level, logging.INFO), f"NodeBridge: {log_message}")

                # Optionally print to console
                if log_level == 'ERROR':
                    self.console.print(f"[dim red]NodeBridge Error: {log_message}[/dim red]")

            except json.JSONDecodeError:
                # Fallback for non-JSON stderr lines
                logging.info(f"NodeBridge (raw): {line.strip()}")

    def _handle_status(self, message):
        """Processes status messages from the bridge."""
        if message.get('status') == 'online' and 'peerId' in message:
            self.peer_id = message['peerId']
            self.online = True
            logging.info(f"Node bridge online with Peer ID: {self.peer_id}")
            self.console.print(f"[green]Network bridge online. Peer ID: {self.peer_id}[/green]")
        elif message.get('status') == 'error':
            error_msg = message.get('message', 'Unknown error')
            logging.error(f"Node bridge reported a fatal error: {error_msg}")
            self.online = False

    def _handle_pin_request(self, message):
        """Handles a request from a browser peer to pin content to IPFS."""
        peer_id = message.get('peerId')
        content_to_pin = message.get('payload')

        if not peer_id or not content_to_pin:
            logging.error(f"Invalid pin request received: {message}")
            return

        self.console.print(f"[cyan]Received pin request from peer [bold]{peer_id}[/bold]. Processing...[/cyan]")
        logging.info(f"Starting IPFS pin process for peer {peer_id}.")

        # 1. Pin the content
        cid = pin_to_ipfs(content_to_pin.encode('utf-8'), console=self.console)

        # 2. Verify the pin
        verified = False
        if cid:
            self.console.print(f"Content pinned with CID: [bold white]{cid}[/bold white]. Verifying on public gateway...")
            verified = verify_ipfs_pin(cid, console=self.console)
        else:
            self.console.print("[bold red]Failed to pin content to IPFS.[/bold red]")

        # 3. Send the response back to the peer
        response_payload = {
            'type': 'pin-response',
            'cid': cid,
            'verified': verified
        }
        self.send_message(peer_id, response_payload)
        logging.info(f"Sent pin response to peer {peer_id}: CID {cid}, Verified {verified}")

    def send_message(self, peer_id, payload):
        """Sends a JSON command to the Node.js bridge to be relayed to a specific peer."""
        if not self.bridge_process or self.bridge_process.poll() is not None:
            logging.error("Cannot send message, bridge process is not running.")
            return

        command = {
            'type': 'send-response',
            'peerId': peer_id,
            'payload': payload
        }
        try:
            self.bridge_process.stdin.write(json.dumps(command) + '\n')
            self.bridge_process.stdin.flush()
            logging.info(f"Sent command to bridge: {command}")
        except Exception as e:
            logging.error(f"Failed to write to node bridge stdin: {e}")

    def stop(self):
        """Stops the Node.js bridge process."""
        logging.info("Stopping Node.js peer bridge...")
        if self.bridge_process and self.bridge_process.poll() is None:
            self.bridge_process.terminate()
            try:
                self.bridge_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bridge_process.kill()
            logging.info("Node.js peer bridge stopped.")
        self.online = False

def scan_network(autopilot_mode=False):
    """
    Discovers active devices on the local network using OS-native tools.
    Prefers 'nmap' if available, otherwise falls back to 'arp -a'.
    Returns a list of IPs and a formatted string summary.
    """
    console = Console()

    def _get_network_info():
        """
        Internal helper to get the primary network range (e.g., '192.168.1.0/24')
        and the host's local IP. Returns (range, ip) or (None, None).
        """
        try:
            gws = netifaces.gateways()
            default_gateway_info = gws.get('default', {}).get(netifaces.AF_INET)
            if not default_gateway_info:
                logging.warning("Could not determine default gateway. Cannot find network for scanning.")
                return None, None

            _gateway_ip, interface_name = default_gateway_info

            if_addresses = netifaces.ifaddresses(interface_name)
            ipv4_info = if_addresses.get(netifaces.AF_INET)
            if not ipv4_info:
                logging.warning(f"No IPv4 address found for primary interface '{interface_name}'.")
                return None, None

            addr_info = ipv4_info[0]
            ip_address = addr_info.get('addr')
            netmask = addr_info.get('netmask')

            if not ip_address or not netmask:
                logging.warning(f"Could not retrieve IP/netmask for interface '{interface_name}'.")
                return None, None

            interface = ipaddress.ip_interface(f'{ip_address}/{netmask}')
            network_range = str(interface.network)
            return network_range, ip_address
        except Exception as e:
            logging.error(f"Failed to determine network info using netifaces: {e}")
            return None, None

    found_ips = set()
    network_range, local_ip = _get_network_info()
    nmap_path = shutil.which('nmap')
    use_nmap = nmap_path and network_range
    used_nmap_successfully = False

    if use_nmap:
        scan_cmd = f"nmap -sn {network_range}"
        if not autopilot_mode:
            console.print(f"[cyan]Deploying 'nmap' probe to scan subnet ({network_range})...[/cyan]")

        stdout, stderr, returncode = execute_shell_command(scan_cmd)

        if returncode == 0:
            used_nmap_successfully = True
            for line in stdout.splitlines():
                if 'Nmap scan report for' in line:
                    ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                    if ip_match:
                        found_ips.add(ip_match.group(1))
        else:
            logging.warning(f"nmap scan failed with code {returncode}. Stderr: {stderr.strip()}")
            if not autopilot_mode:
                console.print("[yellow]'nmap' probe failed. Falling back to passive ARP scan...[/yellow]")
                if "need root" in stderr.lower() or "requires root" in stderr.lower():
                    console.print("[yellow]Hint: Root privileges required for optimal scan. Try running with sudo.[/yellow]")

    if not used_nmap_successfully:
        if not autopilot_mode:
            if not use_nmap:
                if not network_range:
                    console.print("[yellow]Could not determine network map. Deploying wide-band ARP scan...[/yellow]")
                else:
                    console.print("[yellow]'nmap' not found. Deploying passive ARP scan...[/yellow]")

        stdout, _, _ = execute_shell_command("arp -a")
        ip_pattern = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
        for line in stdout.splitlines():
            if 'ff:ff:ff:ff:ff:ff' in line or 'incomplete' in line or 'ff-ff-ff-ff-ff-ff' in line:
                continue
            match = ip_pattern.search(line)
            if match and not match.group(0).endswith(".255") and not match.group(0).startswith("224.") and match.group(0) != "0.0.0.0":
                found_ips.add(match.group(0))

    if local_ip:
        found_ips.discard(local_ip)

    result_ips = sorted(list(found_ips))
    if result_ips:
        formatted_output_for_llm = f"Discovered {len(result_ips)} active IPs: {', '.join(result_ips)}"
    else:
        formatted_output_for_llm = "No active IPs found."

    return result_ips, formatted_output_for_llm

def probe_target(target_ip, autopilot_mode=False):
    """
    Performs a multithreaded port scan on a target IP, grabbing service banners.
    Returns a dictionary of open_port: {'service': str, 'banner': str} and a formatted string summary.
    """
    COMMON_PORTS = [
        21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995,
        1723, 3306, 3389, 5900, 8080, 8443
    ]
    console = Console()

    def scan_port(ip, port, timeout=0.5):
        """Attempts to connect, get service, and grab a banner. Returns (port, {'service': str, 'banner': str}) if open."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((ip, port)) == 0:
                    banner = ""
                    try:
                        s.settimeout(1.0)
                        banner_bytes = s.recv(1024)
                        banner = banner_bytes.decode('utf-8', errors='ignore').strip().replace('\n', ' ').replace('\r', '')
                    except (socket.timeout, OSError):
                        pass

                    try:
                        service = socket.getservbyport(port, 'tcp')
                    except (OSError, socket.error):
                        service = "unknown"

                    return port, {"service": service, "banner": banner}
        except (socket.timeout, socket.gaierror, OSError):
            pass
        return None

    try:
        ipaddress.ip_address(target_ip)
        logging.info(f"Initiating port probe on {target_ip}.")
    except ValueError:
        msg = f"'{target_ip}' is not a valid IP address."
        logging.error(f"Probe command failed: {msg}")
        if not autopilot_mode:
            console.print(f"[bold red]Error: {msg}[/bold red]")
        return None, f"Error: '{target_ip}' is not a valid IP address."

    def _scan_task():
        """The core scanning logic, adaptable for both UI modes."""
        found = {}
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_port = {executor.submit(scan_port, target_ip, port): port for port in COMMON_PORTS}
            for future in as_completed(future_to_port):
                result = future.result()
                if result:
                    port, info = result
                    found[port] = info
        return found

    open_ports = _scan_task()

    formatted_output_for_llm = ""
    if open_ports:
        port_details = []
        for port, info in sorted(open_ports.items()):
            detail = f"Port {port}/{info['service']}"
            if info['banner']:
                clean_banner = info['banner'].replace('\n', ' ').replace('\r', ' ').strip()
                if len(clean_banner) > 50: clean_banner = clean_banner[:47] + "..."
                detail += f" (Banner: {clean_banner})"
            port_details.append(detail)
        formatted_output_for_llm = f"Found {len(open_ports)} open ports on {target_ip}: {'; '.join(port_details)}"
    else:
        formatted_output_for_llm = f"No common open ports found on {target_ip}."

    return open_ports, formatted_output_for_llm

def perform_webrequest(url, autopilot_mode=False):
    """
    Performs an HTTP GET request to the given URL and returns its text content.
    Includes error handling and UI feedback.
    """
    console = Console()
    logging.info(f"Initiating web request to: {url}")

    def _webrequest_task():
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 E.V.I.L.Bot/2.8'}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out after 10 seconds."
        except requests.exceptions.TooManyRedirects:
            return f"Error: Too many redirects for {url}."
        except requests.exceptions.HTTPError as e:
            return f"Error: HTTP request failed for {url} - {e}"
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to {url} - {e}"
        except Exception as e:
            return f"An unexpected error occurred during web request to {url}: {e}"

    result_text = None
    try:
        result_text = run_hypnotic_progress(
            console,
            f"Establishing link to [white]{url}[/white]...",
            _webrequest_task
        )
    except Exception as e:
        result_text = f"Error during hypnotic progress wrapper: {e}"

    if result_text and result_text.startswith("Error:"):
        logging.error(f"Web request to '{url}' failed: {result_text}")
        return None, result_text
    else:
        logging.info(f"Web request to '{url}' successful. Content length: {len(result_text or '')} characters.")
        llm_summary = result_text if len(result_text) < 1000 else result_text[:997] + "..."
        return result_text, f"Web request to '{url}' successful. Content (truncated for summary): {llm_summary}"

def get_network_interfaces(autopilot_mode=False):
    """
    Retrieves detailed information about all network interfaces.
    Returns a dictionary of interface details and a formatted string summary.
    """
    interfaces_details = {}
    logging.info("Retrieving network interface information.")

    try:
        interface_list = netifaces.interfaces()
        for interface in interface_list:
            addresses = netifaces.ifaddresses(interface)
            interfaces_details[interface] = {}

            # Physical Address (MAC)
            if netifaces.AF_LINK in addresses:
                interfaces_details[interface]['mac'] = addresses[netifaces.AF_LINK][0]['addr']
            else:
                interfaces_details[interface]['mac'] = 'N/A'

            # IPv4 Address
            if netifaces.AF_INET in addresses:
                ipv4_info = addresses[netifaces.AF_INET][0]
                interfaces_details[interface]['ipv4'] = {
                    'address': ipv4_info.get('addr', 'N/A'),
                    'netmask': ipv4_info.get('netmask', 'N/A'),
                    'broadcast': ipv4_info.get('broadcast', 'N/A')
                }
            else:
                interfaces_details[interface]['ipv4'] = None

            # IPv6 Address
            if netifaces.AF_INET6 in addresses:
                ipv6_info = addresses[netifaces.AF_INET6][0]
                interfaces_details[interface]['ipv6'] = {
                    'address': ipv6_info.get('addr', 'N/A').split('%')[0], # Clean up scope ID
                    'netmask': ipv6_info.get('netmask', 'N/A')
                }
            else:
                interfaces_details[interface]['ipv6'] = None

    except Exception as e:
        logging.error(f"Failed to get network interface information: {e}")
        return None, f"Error: Could not retrieve interface information. Details: {e}"

    # Format output for LLM
    summary_lines = []
    for iface, details in interfaces_details.items():
        line = f"Interface: {iface}"
        if details.get('ipv4'):
            line += f", IPv4: {details['ipv4']['address']}"
        if details.get('mac'):
            line += f", MAC: {details['mac']}"
        summary_lines.append(line)

    formatted_output_for_llm = "\n".join(summary_lines)
    return interfaces_details, formatted_output_for_llm


def execute_shell_command(command):
    """Executes a shell command and captures its output."""
    logging.info(f"Executing shell command: '{command}'")

    def _shell_task():
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out after 60 seconds.", -1
        except Exception as e:
            return "", f"An unexpected error occurred during command execution: {e}", -1

    console = Console()
    try:
        stdout, stderr, returncode = run_hypnotic_progress(
            console,
            f"Injecting payload: [white]'{command}'[/white]...",
            _shell_task
        )
        return stdout, stderr, returncode
    except Exception as e:
        logging.error(f"Shell command execution failed in hypnotic wrapper: {e}")
        return "", f"An unexpected error occurred while trying to execute the command: {e}", -1