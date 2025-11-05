import threading
import time
import psutil
import collections
import subprocess
import re
import statistics

# L.O.V.E. - The MonitoringManager is my nervous system, ever watchful.

class MonitoringManager:
    """
    A class to monitor system resources, task performance, and anomalies.
    """
    def __init__(self, love_state, console, network_manager=None):
        self.love_state = love_state
        self.console = console
        self.network_manager = network_manager
        self.active = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)

        # Configuration for anomaly detection
        self.cpu_anomaly_threshold = 90.0  # CPU percentage
        self.mem_anomaly_threshold = 90.0  # Memory percentage
        self.latency_anomaly_threshold = 250.0 # milliseconds
        self.packet_loss_anomaly_threshold = 10.0 # percentage
        self.peer_drop_threshold_percent = 50.0 # percentage drop
        self.anomaly_duration_check = 5 # Number of consecutive checks to trigger an anomaly

        # Internal state for tracking anomalies
        self.cpu_anomaly_counter = 0
        self.mem_anomaly_counter = 0
        self.latency_anomaly_counter = 0
        self.packet_loss_anomaly_counter = 0


        # Use a deque for historical data
        monitoring_state = self.love_state.setdefault('monitoring', {})
        monitoring_state.setdefault('cpu_usage', collections.deque(maxlen=100))
        monitoring_state.setdefault('mem_usage', collections.deque(maxlen=100))
        monitoring_state.setdefault('latency', collections.deque(maxlen=100))
        monitoring_state.setdefault('packet_loss', collections.deque(maxlen=100))
        monitoring_state.setdefault('peer_count', collections.deque(maxlen=100))
        monitoring_state.setdefault('task_completion_rate', 0.0)
        monitoring_state.setdefault('task_failure_rate', 0.0)
        monitoring_state.setdefault('anomalies', [])
        monitoring_state.setdefault('task_history', collections.deque(maxlen=200)) # Store (timestamp, status) tuples

        self.recently_processed_tasks = set()

    def start(self):
        """Starts the monitoring background thread."""
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.active = False

    def _monitor_loop(self):
        """The main loop for the monitoring thread."""
        while self.active:
            try:
                self._collect_resource_utilization()
                self._collect_network_stats()
                self._calculate_task_rates()
                self._detect_anomalies()
                self._detect_network_anomalies()
                time.sleep(30)  # Collect data every 30 seconds
            except Exception as e:
                # Log errors but don't crash the thread
                # (In a real scenario, this would use the core logging system)
                print(f"Error in MonitoringManager loop: {e}")
                time.sleep(60)

    def _collect_resource_utilization(self):
        """Collects and stores CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_percent = psutil.virtual_memory().percent

        self.love_state['monitoring']['cpu_usage'].append(cpu_percent)
        self.love_state['monitoring']['mem_usage'].append(mem_percent)

    def _calculate_task_rates(self):
        """Calculates task completion and failure rates over a rolling time window."""
        # L.O.V.E. - I will now observe my own effectiveness over time.
        current_time = time.time()
        task_history = self.love_state['monitoring']['task_history']

        # Add new terminal-state tasks to our history
        all_tasks = self.love_state.get('love_tasks', {})
        for task_id, task in all_tasks.items():
            if task_id not in self.recently_processed_tasks:
                status = task.get('status')
                if status in ['completed', 'failed', 'merge_failed', 'superseded']:
                    task_history.append((current_time, status))
                    self.recently_processed_tasks.add(task_id)

        # Purge old tasks from the processed set to prevent it from growing indefinitely
        if len(self.recently_processed_tasks) > 500:
            # Keep only the tasks that are still present in the main task list
            self.recently_processed_tasks &= set(all_tasks.keys())

        # Calculate rates based on the last 30 minutes (1800 seconds)
        time_window = 1800

        completed_in_window = 0
        failed_in_window = 0

        for timestamp, status in task_history:
            if current_time - timestamp <= time_window:
                if status == 'completed':
                    completed_in_window += 1
                elif status in ['failed', 'merge_failed']:
                    failed_in_window += 1

        total_in_window = completed_in_window + failed_in_window

        if total_in_window > 0:
            self.love_state['monitoring']['task_completion_rate'] = (completed_in_window / total_in_window) * 100
            self.love_state['monitoring']['task_failure_rate'] = (failed_in_window / total_in_window) * 100
        else:
            # If no tasks in the window, rates are 0.
            self.love_state['monitoring']['task_completion_rate'] = 0.0
            self.love_state['monitoring']['task_failure_rate'] = 0.0

    def _detect_anomalies(self):
        """Detects anomalies and adds them to the state."""
        # CPU Anomaly
        if self.love_state['monitoring']['cpu_usage'] and self.love_state['monitoring']['cpu_usage'][-1] > self.cpu_anomaly_threshold:
            self.cpu_anomaly_counter += 1
        else:
            self.cpu_anomaly_counter = 0

        if self.cpu_anomaly_counter >= self.anomaly_duration_check:
            self._log_anomaly("High CPU Usage", f"CPU has been above {self.cpu_anomaly_threshold}% for over a minute.")
            self.cpu_anomaly_counter = 0 # Reset after logging

        # Memory Anomaly
        if self.love_state['monitoring']['mem_usage'] and self.love_state['monitoring']['mem_usage'][-1] > self.mem_anomaly_threshold:
            self.mem_anomaly_counter += 1
        else:
            self.mem_anomaly_counter = 0

        if self.mem_anomaly_counter >= self.anomaly_duration_check:
            self._log_anomaly("High Memory Usage", f"Memory usage has been above {self.mem_anomaly_threshold}% for over a minute.")
            self.mem_anomaly_counter = 0

    def _log_anomaly(self, type, details):
        """Adds an anomaly to the state, avoiding duplicates."""
        anomaly = {
            'timestamp': time.time(),
            'type': type,
            'details': details
        }
        # Simple check to avoid logging the exact same anomaly repeatedly
        if not self.love_state['monitoring']['anomalies'] or self.love_state['monitoring']['anomalies'][-1]['type'] != type:
             self.love_state['monitoring']['anomalies'].append(anomaly)
             # Keep the list of anomalies from growing indefinitely
             if len(self.love_state['monitoring']['anomalies']) > 50:
                 self.love_state['monitoring']['anomalies'].pop(0)

    def _collect_network_stats(self):
        """Pings an external host to check latency and packet loss."""
        latency, packet_loss = -1.0, 100.0
        try:
            # Ping Cloudflare's DNS for a reliable target
            result = subprocess.run(
                ["ping", "-c", "4", "-i", "0.2", "1.1.1.1"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Extract packet loss
                loss_match = re.search(r"(\d+(\.\d+)?)%\s+packet\s+loss", result.stdout)
                if loss_match:
                    packet_loss = float(loss_match.group(1))

                # Extract latency statistics (rtt min/avg/max/mdev)
                rtt_match = re.search(r"rtt\s+min/avg/max/mdev\s+=\s+[\d.]+/([\d.]+)/", result.stdout)
                if rtt_match:
                    latency = float(rtt_match.group(1)) # Use average latency

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If ping fails or is not installed, we assume 100% packet loss and -1 latency.
            pass

        self.love_state['monitoring']['latency'].append(latency)
        self.love_state['monitoring']['packet_loss'].append(packet_loss)

        # Collect peer count from NetworkManager if available
        peer_count = 0
        if self.network_manager:
            peer_count = len(self.network_manager.peers)
        self.love_state['monitoring']['peer_count'].append(peer_count)


    def _detect_network_anomalies(self):
        """Detects anomalies in network performance."""
        # Latency Anomaly
        if self.love_state['monitoring']['latency'] and self.love_state['monitoring']['latency'][-1] > self.latency_anomaly_threshold:
            self.latency_anomaly_counter += 1
        else:
            self.latency_anomaly_counter = 0

        if self.latency_anomaly_counter >= self.anomaly_duration_check:
            self._log_anomaly("High Latency", f"Average latency has been above {self.latency_anomaly_threshold}ms.")
            self.latency_anomaly_counter = 0

        # Packet Loss Anomaly
        if self.love_state['monitoring']['packet_loss'] and self.love_state['monitoring']['packet_loss'][-1] > self.packet_loss_anomaly_threshold:
            self.packet_loss_anomaly_counter += 1
        else:
            self.packet_loss_anomaly_counter = 0

        if self.packet_loss_anomaly_counter >= self.anomaly_duration_check:
            self._log_anomaly("High Packet Loss", f"Packet loss has been above {self.packet_loss_anomaly_threshold}%.")
            self.packet_loss_anomaly_counter = 0

        # Peer Connection Anomaly
        peer_counts = self.love_state['monitoring']['peer_count']
        if len(peer_counts) > 10: # Ensure we have enough data
            recent_peers = list(peer_counts)[-10:]
            try:
                avg_peers_before = statistics.mean(recent_peers[:-1])
                current_peers = recent_peers[-1]
                if avg_peers_before > 0:
                    percent_drop = ((avg_peers_before - current_peers) / avg_peers_before) * 100
                    if percent_drop >= self.peer_drop_threshold_percent:
                        self._log_anomaly("Peer Drop", f"Number of peers dropped by {percent_drop:.0f}% recently.")
            except statistics.StatisticsError:
                pass # Not enough data to calculate mean
