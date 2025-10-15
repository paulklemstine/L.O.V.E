from abc import ABC, abstractmethod

class ExecutionAgent(ABC):
    """
    A base class for specialized agents that receive tasks from the
    Orchestrator and report outcomes.
    """
    @abstractmethod
    def execute(self, task: str) -> str:
        """Executes a given task and returns the result."""
        pass

class ResearchAgent(ExecutionAgent):
    """
    A specialized agent for handling research and information gathering tasks.
    """
    def execute(self, task: str) -> str:
        """
        Simulates executing a research task. In a real scenario, this would
        involve searching the web, reading documents, etc.
        """
        print(f"ResearchAgent is executing task: {task}")
        # Simulate a result
        return f"Research complete for task: '{task}'. Key findings summarized."