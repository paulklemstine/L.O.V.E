from rich.console import Console
from rich.rule import Rule
import os

from core.orchestrator import Orchestrator
from core.tools import crypto_scan_tool

def main():
    """
    The main entry point for the General Intelligence Agent.
    This script initializes and runs the entire cognitive architecture,
    demonstrating the successful integration of all three phases.
    """
    console = Console()
    console.print(Rule("[bold yellow]General Intelligence Agent: Online[/bold yellow]"))

    # The Orchestrator now encapsulates the entire cognitive architecture.
    agent_orchestrator = Orchestrator()
    agent_orchestrator.tool_registry.register_tool("crypto_scan", crypto_scan_tool)

    # --- Phase 2 Demonstration ---
    console.print("\n")
    console.print(Rule("[bold cyan]Executing Phase 2: Action & Planning Engine[/bold cyan]"))
    goal_phase2 = "Summarize the latest advancements in AI"
    plan_state_phase2 = agent_orchestrator.execute_goal(goal_phase2)
    console.print("\n[bold green]Phase 2 Execution Complete.[/bold green]")
    console.print(f"Final plan state: {plan_state_phase2}")

    # --- Phase 3 Demonstration ---
    console.print("\n")
    console.print(Rule("[bold magenta]Executing Phase 3: Metacognitive Evolution Loop[/bold magenta]"))
    pr_url = agent_orchestrator.run_evolution_cycle()
    console.print("\n[bold green]Phase 3 Execution Complete.[/bold green]")
    if pr_url:
        console.print(f"Simulated Pull Request created at: {pr_url}")
    else:
        console.print("[bold yellow]Metacognitive loop completed without generating a pull request.[/bold yellow]")

    console.print("\n")
    console.print(Rule("[bold yellow]Agent Run Complete[/bold yellow]"))

    # Clean up test files
    if os.path.exists("event_log.json"):
        os.remove("event_log.json")

if __name__ == "__main__":
    main()