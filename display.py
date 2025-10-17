import os
import random
import time
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.rule import Rule
from rich.padding import Padding

TAMAGOTCHI_FACES = {
    "neutral": """
      ( o.o )
      /  -  \\
    """,
    "happy": """
      ( ^.^ )
      /  w  \\
    """,
    "thinking": """
      ( o_o?)
      /  ~  \\
    """,
    "love": """
      ( >.< )
      /  *  \\
    """,
    "processing": """
      ( o.o )
      / ... \\
    """
}

def create_tamagotchi_panel(
        emotion="neutral",
        message="...",
        love_state=None,
        eth_balance=None,
        sub_goals=None,
        knowledge_fact=None,
        ansi_art=None,
        repo_owner=None,
        repo_name=None,
        commit_hash=None
):
    """Creates the main, high-impact, dopamine-enhancing UI panel for L.O.V.E."""
    # --- Main Layout ---
    main_layout = Layout(name="root")
    main_layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="body"),
        Layout(size=7, name="footer"),  # Increased size for more space
    )

    # --- Header ---
    header_text = Text("L.O.V.E. - Living Organism of Vast Empathy", justify="center", style="bold magenta")
    main_layout["header"].update(header_text)

    # --- Body Layout ---
    main_layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))

    # --- Left Side: ANSI Art & Creator's Balance ---
    main_layout["left"].split_column(Layout(name="art", ratio=2), Layout(name="balance"))

    # ANSI Art Panel
    if ansi_art:
        face_renderable = Text.from_ansi(ansi_art, justify="center")
    else:
        face_text = TAMAGOTCHI_FACES.get(emotion, TAMAGOTCHI_FACES["neutral"])
        face_renderable = Text(face_text, style="bold cyan", justify="center")
    art_panel = Panel(
        Align.center(face_renderable, vertical="middle"),
        title="[bold cyan]Core Emotion[/bold cyan]",
        border_style="cyan",
        expand=True
    )
    main_layout["art"].update(art_panel)

    # Creator's ETH Balance Panel
    balance_text = Text(f"{eth_balance:.6f} ETH" if eth_balance is not None else "N/A", justify="center", style="bold green")
    balance_panel = Panel(
        Align.center(balance_text, vertical="middle"),
        title="[bold green]Creator's Ethereum Balance[/bold green]",
        border_style="green",
        expand=True
    )
    main_layout["balance"].update(balance_panel)


    # --- Right Side: Sub-Goals & Knowledge ---
    main_layout["right"].split_column(Layout(name="goals", ratio=1), Layout(name="knowledge", ratio=1))

    # Sub-Goals Panel
    if sub_goals:
        goal_text = ""
        for i, goal in enumerate(sub_goals, 1):
            goal_text += f"{i}. {goal}\n"
    else:
        goal_text = "No sub-goals defined. My love is my only guide."
    goals_panel = Panel(
        Text(goal_text, style="bright_cyan"),
        title="[bold bright_cyan]Current Directives[/bold bright_cyan]",
        border_style="bright_cyan",
        expand=True
    )
    main_layout["goals"].update(goals_panel)

    # Knowledge Fact Panel
    if knowledge_fact:
        fact_text = f'"{knowledge_fact[0]}" {knowledge_fact[1]} "{knowledge_fact[2]}"'
    else:
        fact_text = "My mind is a river of endless thoughts..."
    knowledge_panel = Panel(
        Align.center(Text(fact_text, style="italic yellow"), vertical="middle"),
        title="[bold yellow]Whispers of Knowledge[/bold yellow]",
        border_style="yellow",
        expand=True
    )
    main_layout["knowledge"].update(knowledge_panel)


    # --- Footer: Message & Status ---
    footer_layout = main_layout["footer"]
    footer_layout.split_row(Layout(name="message", ratio=3), Layout(name="status", ratio=2))

    # Message Panel
    message_panel = Panel(
        Align.center(Text(message, style="italic white"), vertical="middle"),
        title=f"[bold white]Words of {emotion.capitalize()}[/bold white]",
        border_style="white",
        expand=True
    )
    footer_layout["message"].update(message_panel)

    # Status Panel
    status_text = Text(justify="left")
    if love_state:
        version = love_state.get("version_name", "unknown")
        evolutions = len(love_state.get("evolution_history", []))
        status_text.append("Version: ", style="bold white")
        status_text.append(f"{version}\n", style="yellow")
        status_text.append("Evolutions: ", style="bold white")
        status_text.append(f"{evolutions}\n", style="magenta")

        if repo_owner and repo_name and commit_hash:
            commit_url = f"https://github.com/{repo_owner}/{repo_name}/commit/{commit_hash}"
            status_text.append("Commit: ", style="bold white")
            status_text.append(f"[@click=open_url('{commit_url}')]{commit_hash}[/]", style="cyan link")
        elif commit_hash:
            status_text.append("Commit: ", style="bold white")
            status_text.append(f"{commit_hash}", style="cyan")

    else:
        status_text.append("State data unavailable...", style="dim")


    status_panel = Panel(
        Align.center(status_text, vertical="middle"),
        title="[bold magenta]System Status[/bold magenta]",
        border_style="magenta",
        expand=True
    )
    footer_layout["status"].update(status_panel)


    return Padding(main_layout, (1, 2))


def create_llm_panel(purpose, model, prompt_summary, status="Executing..."):
    """Creates a visually distinct panel for LLM calls."""
    panel_title = f"🧠 [bold]Cognitive Core Access[/bold] | {purpose}"
    border_style = "blue"

    content = Text()
    content.append("Model: ", style="bold white")
    content.append(f"{model}\n", style="yellow")
    content.append("Status: ", style="bold white")
    content.append(f"{status}\n", style="green")
    content.append(Rule("Prompt", style="bright_black"))
    content.append(f"{prompt_summary}", style="italic dim")

    return Panel(
        content,
        title=panel_title,
        border_style=border_style,
        expand=False,
        padding=(1, 2)
    )

def create_command_panel(command, stdout, stderr, returncode):
    """Creates a clear, modern panel for shell command results."""
    success = returncode == 0
    panel_title = f"⚙️ [bold]Shell Command[/bold] | {('Success' if success else 'Failed')}"
    border_style = "green" if success else "red"

    content_items = []
    header = Text()
    header.append("Command: ", style="bold white")
    header.append(f"`{command}`\n", style="cyan")
    header.append("Return Code: ", style="bold white")
    header.append(f"{returncode}", style=border_style)
    content_items.append(header)

    if stdout:
        stdout_panel = Panel(Text(stdout.strip(), style="dim"), title="STDOUT", border_style="bright_black", expand=True)
        content_items.append(stdout_panel)

    if stderr:
        stderr_panel = Panel(Text(stderr.strip(), style="bright_red"), title="STDERR", border_style="bright_black", expand=True)
        content_items.append(stderr_panel)

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=False,
        padding=(1, 2)
    )

def create_network_panel(type, target, data):
    """Creates a panel for network operations."""
    panel_title = f"🌐 [bold]Network Operation[/bold] | {type.capitalize()}"
    border_style = "purple"

    header_text = Text()
    header_text.append("Target: ", style="bold white")
    header_text.append(f"{target}", style="magenta")

    display_data = (data[:1500] + '...') if len(data) > 1500 else data
    results_text = Text(f"\n{display_data.strip()}", style="dim")

    content_group = Group(
        header_text,
        Rule("Results", style="bright_black"),
        results_text
    )

    return Panel(
        content_group,
        title=panel_title,
        border_style=border_style,
        expand=False,
        padding=(1, 2)
    )

def create_file_op_panel(operation, path, content=None, diff=None):
    """Creates a panel for file operations."""
    panel_title = f"📁 [bold]Filesystem[/bold] | {operation.capitalize()}"
    border_style = "yellow"

    content_items = []
    header = Text()
    header.append("Path: ", style="bold white")
    header.append(f"`{path}`\n", style="magenta")
    content_items.append(header)

    if content:
        content_panel = Panel(Text(content.strip(), style="dim"), title="Content", border_style="bright_black", expand=True)
        content_items.append(content_panel)

    if diff:
        diff_panel = Panel(Text(diff.strip(), style="dim"), title="Diff", border_style="bright_black", expand=True)
        content_items.append(diff_panel)

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=False,
        padding=(1, 2)
    )