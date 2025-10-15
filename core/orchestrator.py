class Orchestrator:
    def __init__(self, execution_agent):
        self.execution_agent = execution_agent

    def execute_task(self, task):
        print(f"Orchestrator delegating task: {task}")
        result = self.execution_agent.execute(task)
        print(f"Orchestrator received result: {result}")
        return result