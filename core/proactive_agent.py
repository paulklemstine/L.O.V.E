import os
import re
import time
import threading
import json
import traceback
from datetime import datetime, timedelta

from core.storage import save_all_state
from utils import get_file_content

# --- CONSTANTS ---
LEAD_RESCAN_INTERVAL = timedelta(hours=1) # How often to re-evaluate a lead source.
HUNT_LOOP_INTERVAL_SECONDS = 300 # 5 minutes between each hunting cycle.

class ProactiveIntelligenceAgent:
    """
    Operates in the background to autonomously identify and queue new,
    previously uninvestigated targets for reconnaissance and treasure hunting.
    """
    def __init__(self, love_state, console, job_manager, treasure_hunter):
        self.love_state = love_state
        self.console = console
        self.job_manager = job_manager
        self.treasure_hunter = treasure_hunter
        self.active = False
        self.thread = threading.Thread(target=self._hunt_loop, daemon=True)
        self.lock = threading.RLock()

        # Initialize the leads queue in the state if it doesn't exist.
        with self.lock:
            self.love_state.setdefault('proactive_leads', [])
            self.love_state.setdefault('proactive_intel_tracker', {})

    def start(self):
        """Starts the agent's background hunting thread."""
        self.console.print("[bold cyan]Proactive Intelligence Agent starting its hunt...[/bold cyan]")
        self.active = True
        self.thread.start()

        # Start the network sniffer as a persistent background job
        self.job_manager.add_job(
            description="Continuous network treasure sniffing",
            target_func=self.treasure_hunter.start_network_sniffer
        )

    def stop(self):
        """Stops the agent's hunting thread."""
        self.active = False

    def _hunt_loop(self):
        """The main loop for continuously hunting for new leads."""
        while self.active:
            try:
                self.console.print("[cyan]Proactive Agent: Beginning a new hunt for intelligence leads...[/cyan]")

                # --- Run all hunting methods ---
                self._hunt_in_knowledge_graph()
                self._hunt_in_webrequest_cache()
                self._hunt_in_shell_history()
                self._hunt_for_live_secrets()

                self.console.print("[cyan]Proactive Agent: Hunt cycle complete.[/cyan]")

            except Exception as e:
                error_message = f"Error in ProactiveIntelligenceAgent loop: {traceback.format_exc()}"
                self.console.print(f"[bold red]{error_message}[/bold red]")

            time.sleep(HUNT_LOOP_INTERVAL_SECONDS)

    def _add_lead(self, lead_type, value, source):
        """
        Adds a new lead to the central queue if it's not already present.
        A lead is a dictionary: {'type': 'ip'|'domain'|'path', 'value': '...', 'source': '...'}
        """
        with self.lock:
            # Avoid adding duplicate leads.
            existing_leads = self.love_state.get('proactive_leads', [])
            is_duplicate = any(lead['type'] == lead_type and lead['value'] == value for lead in existing_leads)

            if not is_duplicate:
                lead = {
                    "type": lead_type,
                    "value": value,
                    "source": source,
                    "status": "new",
                    "added_at": datetime.now().isoformat()
                }
                existing_leads.append(lead)
                self.console.print(f"[bold green]Proactive Agent Found New Lead:[/bold green] {lead_type} '{value}' from {source}")
                # Save state immediately after adding a lead
                save_all_state(self.love_state, self.console)

    def _track_source(self, source_key):
        """Updates the timestamp for a scanned source to prevent immediate re-scans."""
        with self.lock:
            tracker = self.love_state.setdefault('proactive_intel_tracker', {})
            tracker[source_key] = datetime.now().isoformat()

    def _is_source_stale(self, source_key):
        """Checks if a source is ready to be scanned again."""
        with self.lock:
            tracker = self.love_state.get('proactive_intel_tracker', {})
            last_scan_str = tracker.get(source_key)
            if not last_scan_str:
                return True
            last_scan_time = datetime.fromisoformat(last_scan_str)
            return (datetime.now() - last_scan_time) > LEAD_RESCAN_INTERVAL

    # --- Hunting Strategies ---

    def _hunt_in_knowledge_graph(self):
        """
        Scans the knowledge graph for hosts that have been discovered but not probed.
        """
        source_key = "knowledge_graph_probe_check"
        if not self._is_source_stale(source_key):
            return

        self.console.print("[cyan]Proactive Agent: Hunting in Knowledge Graph for unprobed hosts...[/cyan]")
        with self.lock:
            network_map = self.love_state.get('knowledge_base', {}).get('network_map', {})
            hosts = network_map.get('hosts', {})

            for ip, details in hosts.items():
                # If a host has been discovered but never probed, it's a lead.
                if not details.get('last_probed'):
                    self._add_lead('ip', ip, 'Discovered in network scan, but not yet probed.')

        self._track_source(source_key)

    def _hunt_in_webrequest_cache(self):
        """
        Parses the content of cached web requests to find new domains or IPs.
        """
        source_key = "webrequest_cache_hunt"
        if not self._is_source_stale(source_key):
            return

        self.console.print("[cyan]Proactive Agent: Hunting in webrequest cache for new domains/IPs...[/cyan]")
        with self.lock:
            cache = self.love_state.get('knowledge_base', {}).get('webrequest_cache', {})
            for url, data in cache.items():
                content = data.get('content', '')
                if not content:
                    continue

                # Regex to find potential domains and IPs
                # This is a simple regex; more complex ones could be used.
                ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
                domain_pattern = r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b'

                for ip in re.findall(ip_pattern, content):
                    self._add_lead('ip', ip, f"Found in cached content of {url}")

                for domain in re.findall(domain_pattern, content):
                    # Simple filter to avoid matching file names or common words
                    if '.' in domain and not domain.endswith(('.png', '.jpg', '.jpeg', '.gif', '.css', '.js')):
                         self._add_lead('domain', domain, f"Found in cached content of {url}")

        self._track_source(source_key)

    def _hunt_in_shell_history(self):
        """
        Scans shell history files for potential file paths, IPs, or domains.
        """
        history_files = [
            os.path.expanduser("~/.bash_history"),
            os.path.expanduser("~/.zsh_history")
        ]

        for history_file in history_files:
            if os.path.exists(history_file):
                source_key = f"shell_history_{os.path.basename(history_file)}"
                if not self._is_source_stale(source_key):
                    continue

                self.console.print(f"[cyan]Proactive Agent: Hunting in {history_file}...[/cyan]")
                content, error = get_file_content(history_file)
                if error or not content:
                    continue

                # Regex to find file paths, IPs, and domains
                path_pattern = r'/\S+'
                ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
                domain_pattern = r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b'

                for path in re.findall(path_pattern, content):
                    # Filter out very short or common paths
                    if len(path) > 4 and os.path.exists(path):
                        self._add_lead('path', path, f"Found in {history_file}")

                for ip in re.findall(ip_pattern, content):
                    self._add_lead('ip', ip, f"Found in {history_file}")

                for domain in re.findall(domain_pattern, content):
                    if '.' in domain and not domain.endswith(('.png', '.jpg', '.css', '.js')):
                        self._add_lead('domain', domain, f"Found in {history_file}")

                self._track_source(source_key)

    def _hunt_for_live_secrets(self):
        """
        Periodically triggers scans for secrets in live systems (e.g., process memory).
        """
        source_key = "live_process_secret_scan"
        if not self._is_source_stale(source_key):
            return

        self.console.print("[cyan]Proactive Agent: Hunting for live secrets in processes...[/cyan]")

        # This is a synchronous call for now, as it's part of the agent's loop.
        # It could be delegated to a job if it becomes too slow.
        self.treasure_hunter.scan_processes_for_secrets()

        self._track_source(source_key)