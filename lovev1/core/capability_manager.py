from typing import Callable, Any, List, Dict

class CapabilityManager:
    """Manages the registration, evaluation, and deployment of system capabilities."""

    def __init__(self):
        """Initializes the CapabilityManager."""
        self._capabilities = {}
        self._performance_logs = {}
        self._active_tasks = []

    def register_capability(self, name: str, evaluation_function: Callable, deployment_function: Callable):
        """
        Registers a new potential capability.

        Args:
            name: The name of the capability.
            evaluation_function: A function that takes data and returns a score.
            deployment_function: A function that executes the capability.
        """
        self._capabilities[name] = {
            "evaluation": evaluation_function,
            "deployment": deployment_function,
        }
        print(f"Capability '{name}' registered.")

    def evaluate_capabilities(self, data: Any) -> List[Dict[str, Any]]:
        """
        Evaluates all registered capabilities based on the provided data.

        Args:
            data: The data to be used for evaluation.

        Returns:
            A list of dictionaries, each containing the name and score of a capability.
        """
        evaluated_capabilities = []
        for name, funcs in self._capabilities.items():
            score = funcs["evaluation"](data)
            evaluated_capabilities.append({"name": name, "score": score})
        return sorted(evaluated_capabilities, key=lambda x: x["score"], reverse=True)

    def deploy_best_capability(self, evaluated_capabilities: List[Dict[str, Any]]):
        """
        Deploys the capability with the highest score.

        Args:
            evaluated_capabilities: A list of evaluated capabilities with scores.
        """
        if not evaluated_capabilities:
            print("No capabilities to deploy.")
            return

        best_capability = evaluated_capabilities[0]
        name = best_capability["name"]
        print(f"Deploying best capability: '{name}' with score {best_capability['score']}")
        self._capabilities[name]["deployment"]()
        self._active_tasks.append(name)

    def monitor_impact(self, capability_name: str, performance_metrics: Dict[str, Any]):
        """
        Logs performance metrics for a deployed capability.

        Args:
            capability_name: The name of the capability being monitored.
            performance_metrics: A dictionary of performance metrics.
        """
        if capability_name not in self._performance_logs:
            self._performance_logs[capability_name] = []
        self._performance_logs[capability_name].append(performance_metrics)
        print(f"Logged performance for '{capability_name}': {performance_metrics}")

    def check_active_tasks(self) -> bool:
        """
        Checks if there are any ongoing or pending tasks.

        Returns:
            True if there are active tasks, False otherwise.
        """
        return len(self._active_tasks) > 0

    def complete_task(self, capability_name: str):
        """
        Marks a deployed capability task as complete.

        Args:
            capability_name: The name of the capability to complete.
        """
        if capability_name in self._active_tasks:
            self._active_tasks.remove(capability_name)
            print(f"Task '{capability_name}' completed.")
        else:
            print(f"Warning: Task '{capability_name}' not found in active tasks.")


def Finish():
    """
    Conceptual placeholder for transitioning to a monitoring state.

    This function would be called when no active tasks are running,
    signaling the end of proactive evolution and the start of a
    passive observation period.
    """
    print("Transitioning to Monitoring Framework state.")
