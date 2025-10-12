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
        self.console = console
        self.bridge_process = None
        self.peer_id = None
        self.online = False

    def run(self):
        logging.info("Starting Node.js peer bridge...")
        try:
            self.bridge_process = subprocess.Popen(
                ['node', 'peer_bridge.js'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 # Line-buffered
            )
            # Start threads to handle stdout and stderr from the bridge
            Thread(target=self._read_stdout, daemon=True).start()
            Thread(target=self._read_stderr, daemon=True).start()
        except FileNotFoundError:
            logging.error("Node.js is not installed or not in PATH. P2P functionality disabled.")
            if self.console:
                self.console.print("[bold red]Error: 'node' command not found. P2P features will be disabled.[/bold red]")
            self.online = False
        except Exception as e:
            logging.critical(f"Failed to start peer_bridge.js: {e}")
            if self.console:
                self.console.print(f"[bold red]Critical error starting Node.js bridge: {e}[/bold red]")
            self.online = False

    def _read_stdout(self):
        """Reads and processes messages from the Node.js bridge's stdout."""
        for line in iter(self.bridge_process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            logging.debug(f"NodeBridge STDOUT: {line}")
            if line.startswith('PeerJS bridge connected with ID:'):
                self.peer_id = line.split(':')[1].strip()
                self.online = True
                logging.info(f"Node bridge online with Peer ID: {self.peer_id}")
                if self.console:
                    self.console.print(f"[green]Network bridge online. Peer ID: {self.peer_id}[/green]")

    def _read_stderr(self):
        """Logs messages from the Node.js bridge's stderr."""
        for line in iter(self.bridge_process.stderr.readline, ''):
            if not line:
                break
            logging.info(f"NodeBridgeLog: {line.strip()}")
            if "Error" in line or "error" in line:
                 if self.console:
                    self.console.print(f"[bold red]Node Bridge Error: {line.strip()}[/bold red]")


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