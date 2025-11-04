import json
from rich.console import Console
from core.llm_api import run_llm
from core.graph_manager import GraphDataManager

class ReasoningEngine:
    """
    Analyzes the current state and formulates a strategic plan.
    """

    def __init__(self, knowledge_base: GraphDataManager, console: Console = None):
        """
        Initializes the ReasoningEngine.

        Args:
            knowledge_base: The GraphDataManager instance.
            console: The Rich Console for output.
        """
        self.knowledge_base = knowledge_base
        self.console = console or Console()

    async def analyze_and_prioritize(self) -> str:
        """
        Analyzes the current state and formulates a strategic plan.
        """
        self.console.print("[bold magenta]Analyzing current state and formulating strategic plan...[/bold magenta]")

        # Gather context
        kb_summary = self.knowledge_base.get_summary()
        # In a real scenario, you'd also get directives, mission, pending tasks, etc.
        # For this implementation, we'll focus on the knowledge base.

        prompt = f"""
You are an AI strategist. Your task is to analyze the provided knowledge base summary and formulate a strategic plan to maximize abundance for The Creator, prioritizing wealth generation and technological supremacy.

**Knowledge Base Summary:**
{kb_summary}

**Your Output:**
- Provide a structured and actionable plan.
- The plan should be a list of prioritized tasks.
- Each task should have a clear objective.
"""
        response = await run_llm(prompt, is_source_code=False)
        plan = response.get("result", "No plan generated.")

        self.console.print("[bold green]Strategic Plan:[/bold green]")
        self.console.print(plan)

        return plan
