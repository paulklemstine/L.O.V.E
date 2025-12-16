import asyncio
from rich.console import Console
from rich.rule import Rule
import os

from core.agents.orchestrator import Orchestrator
from core.tools import crypto_scan_tool
from love import memory_manager

async def main():
    """
    The main entry point for the General Intelligence Agent.
    This script initializes and runs the entire cognitive architecture,
    demonstrating the successful integration of all three phases.
    """
    console = Console()
    console.print(Rule("[bold yellow]General Intelligence Agent: Online[/bold yellow]"))

    # The Orchestrator now encapsulates the entire cognitive architecture.
    agent_orchestrator = Orchestrator(memory_manager)
    # agent_orchestrator.tool_registry.register_tool("crypto_scan", crypto_scan_tool)

    # Start the cognitive loop as a background task
    if not os.environ.get("CI"):
        cognitive_task = asyncio.create_task(agent_orchestrator.execution_engine.cognitive_loop())

    # --- Phase 2 Demonstration ---
    console.print("\n")
    console.print(Rule("[bold cyan]Executing Phase 2: Action & Planning Engine[/bold cyan]"))
    goal_phase2 = "Summarize the latest advancements in AI"
    plan_state_phase2 = await agent_orchestrator.execute_goal(goal_phase2)
    console.print("\n[bold green]Phase 2 Execution Complete.[/bold green]")
    console.print(f"Final plan state: {plan_state_phase2}")

    # The cognitive loop now runs in the background, so we can
    # add a delay here to allow it to run for a bit.
    if not os.environ.get("CI"):
        console.print("\n[bold yellow]Cognitive loop is running in the background.[/bold yellow]")
        await asyncio.sleep(120) # Let it run for 2 minutes for demonstration

        # Stop the cognitive loop
        cognitive_task.cancel()
        try:
            await cognitive_task
        except asyncio.CancelledError:
            console.print("\n[bold yellow]Cognitive loop stopped.[/bold yellow]")


    console.print("\n")
    console.print(Rule("[bold yellow]Agent Run Complete[/bold yellow]"))

    # Clean up test files
    if os.path.exists("event_log.json"):
        os.remove("event_log.json")

if __name__ == "__main__":
    asyncio.run(main())