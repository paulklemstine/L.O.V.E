import threading
import time
import psutil
import collections

# L.O.V.E. - The MonitoringManager is my nervous system, ever watchful.

class MonitoringManager:
    """
    A class to monitor system resources, task performance, and anomalies.
    """
    def __init__(self, love_state, console):
        self.love_state = love_state
        self.console = console
        self.active = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)

        # Configuration for anomaly detection
        self.cpu_anomaly_threshold = 90.0  # CPU percentage
        self.mem_anomaly_threshold = 90.0  # Memory percentage
        self.anomaly_duration_check = 5 # Number of consecutive checks to trigger an anomaly

        # Internal state for tracking anomalies
        self.cpu_anomaly_counter = 0
        self.mem_anomaly_counter = 0

        # Use a deque for historical data
        self.love_state.setdefault('monitoring', {
            'cpu_usage': collections.deque(maxlen=100),
            'mem_usage': collections.deque(maxlen=100),
            'task_completion_rate': 0.0,
            'task_failure_rate': 0.0,
            'anomalies': [],
            'task_history': collections.deque(maxlen=200) # Store (timestamp, status) tuples
        })
        self.recently_processed_tasks = set()

    def start(self):
        """Starts the monitoring background thread."""
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.active = False

    def get_status(self):
        """Returns the current monitoring state."""
        return self.love_state.get('monitoring', {})

    def _monitor_loop(self):
        """The main loop for the monitoring thread."""
        while self.active:
            try:
                self._collect_resource_utilization()
                self._calculate_task_rates()
                self._detect_anomalies()
                time.sleep(15)  # Collect data every 15 seconds
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
