from .execution import ExecutionAgent

class Orchestrator:
    """
    The central controller responsible for high-level goal setting,
    task decomposition, and delegation to specialized execution agents.
    """
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_name: str, agent: ExecutionAgent):
        """Registers an execution agent with the orchestrator."""
        self.agents[agent_name] = agent
        print(f"Agent '{agent_name}' registered.")

    def delegate_task(self, agent_name: str, task: str) -> str:
        """Delegates a task to a specified agent."""
        if agent_name in self.agents:
            print(f"Delegating task to '{agent_name}': {task}")
            return self.agents[agent_name].execute(task)
        else:
            return f"Error: Agent '{agent_name}' not found."