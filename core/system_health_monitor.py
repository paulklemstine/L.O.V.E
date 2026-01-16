import asyncio
from core.logging import log_event

class SystemHealthMonitor:
    """
    Monitors the health of core agents and capabilities.
    """

    def __init__(self, agents_to_monitor, check_interval=60):
        self.agents_to_monitor = agents_to_monitor
        self.check_interval = check_interval
        self._is_running = False

    def is_alive(self):
        """
        Health check for the monitor itself.
        """
        return self._is_running

    async def run(self):
        """
        The main loop for the health monitor.
        """
        log_event("SystemHealthMonitor started.", level="INFO")
        self._is_running = True

        while self._is_running:
            for agent_name, agent_instance in self.agents_to_monitor.items():
                if not agent_instance:
                    log_event(f"Health check for {agent_name}: SKIPPED (agent not initialized)", level="WARNING")
                    continue

                try:
                    if hasattr(agent_instance, 'is_alive') and agent_instance.is_alive():
                        log_event(f"Health check for {agent_name}: OK", level="INFO")
                    else:
                        log_event(f"Health check for {agent_name}: FAILED", level="CRITICAL")
                except Exception as e:
                    log_event(f"Health check for {agent_name}: FAILED with exception: {e}", level="CRITICAL")

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """
        Stops the health monitor.
        """
        self._is_running = False
