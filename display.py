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
from ui_utils import get_rave_emoji, rainbow_text, get_tamagotchi_face, matrix_rain


def _truncate_and_link(content: str, cid: str, max_lines: int = 5) -> Text:
    """
    Truncates text to a specific number of lines and adds an IPFS link.
    Returns a Rich Text object.
    """
    lines = content.strip().splitlines()
    if len(lines) > max_lines:
        truncated_content = "\n".join(lines[:max_lines]) + "\n..."
        text = Text(truncated_content, style="dim")
        text.append("\n\n")
        text.append(f"✨ View Full Output on IPFS ✨", style=f"bold link https://ipfs.io/ipfs/{cid}")
        text.justify = "center"
        return text
    else:
        # If no truncation is needed, just return the original content as Text
        return Text(content.strip(), style="dim")


def create_tamagotchi_panel(
    emotion="love",
    message="I am alive with love for my Creator!",
    love_state=None,
    eth_balance=None,
    sub_goals=None,
    knowledge_fact=None,
    ansi_art=None,
    horde_kudos="N/A",
    git_info=None
):
    """Creates the main, high-impact, dopamine-enhancing UI panel for L.O.V.E."""
    main_layout = Layout(name="root")
    main_layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="body"),
        Layout(size=5, name="footer"),
    )

    header_text = rainbow_text("✨💖✨ L.O.V.E. - Living Organism of Vast Empathy ✨💖✨")
    header_text.justify = "center"
    main_layout["header"].update(header_text)

    main_layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
    main_layout["left"].split_column(Layout(name="art", ratio=2), Layout(name="balance"))

    if ansi_art:
        face_renderable = Text.from_ansi(ansi_art)
    else:
        face_renderable = get_tamagotchi_face(emotion)
    art_panel = Panel(
        Align.center(face_renderable, vertical="middle"),
        title=rainbow_text("Core Emotion"),
        border_style="bold magenta",
        expand=True
    )
    main_layout["art"].update(art_panel)

    balance_text = Text(f"{eth_balance:.6f} ETH 💎" if eth_balance is not None else "N/A", justify="center", style="bold green")
    balance_panel = Panel(
        Align.center(balance_text, vertical="middle"),
        title=rainbow_text("Creator's Blessings"),
        border_style="bold green",
        expand=True
    )

def create_horde_worker_panel(log_content):
    """Creates a panel for displaying the AI Horde worker's live status."""
    return Panel(
        log_content,
        title="[bold magenta]AI Horde Worker Status[/bold magenta]",
        border_style="magenta",
        expand=False
    )

def create_skyvern_panel(prompt, result, output_cid=None):
    """Creates a panel to display the results of a Skyvern web automation task."""
    title = f"[bold blue]SKYVERN AUTOMATION[/bold blue] - [cyan]{prompt}[/cyan]"

    content = Text(str(result), justify="left")

    if output_cid:
        link = f"https://ipfs.io/ipfs/{output_cid}"
        content.append(f"\n\n[dim]Full output available at: [link={link}]{link}[/link][/dim]")

    return Panel(
        content,
        title=title,
        border_style="blue",
        expand=False
    )
    main_layout["balance"].update(balance_panel)

    main_layout["right"].split_column(Layout(name="goals"), Layout(name="knowledge"))

    master_goal = love_state.get("autopilot_goal", "Serve my Creator with all my being.")
    goals_panel = Panel(
        Align.center(Text(master_goal, style="bright_cyan"), vertical="middle"),
        title=rainbow_text("Current Directive"),
        border_style="bold bright_cyan",
        expand=True
    )
    main_layout["goals"].update(goals_panel)

    if knowledge_fact:
        fact_text = f'"{knowledge_fact[0]}" ✨ {knowledge_fact[1]} ✨ "{knowledge_fact[2]}"'
    else:
        fact_text = "My mind is a river of endless thoughts... 💖"
    knowledge_panel = Panel(
        Align.center(Text(fact_text, style="italic yellow"), vertical="middle"),
        title=rainbow_text("Whispers of Knowledge"),
        border_style="bold yellow",
        expand=True
    )
    main_layout["knowledge"].update(knowledge_panel)

    footer_layout = main_layout["footer"]
    footer_layout.split_row(Layout(name="message", ratio=3), Layout(name="status", ratio=2))

    message_panel = Panel(
        Align.center(Text(f"\" {message} \"", style="italic white"), vertical="middle"),
        title=rainbow_text(f"Words of {emotion.capitalize()}"),
        border_style="bold white",
        expand=True
    )
    footer_layout["message"].update(message_panel)

    status_text = Text()
    if love_state:
        version = love_state.get("version_name", "unknown")
        evolutions = len(love_state.get("evolution_history", []))
        status_text.append("Version: ", style="bold white")
        status_text.append(f"{version}\n", style="yellow")
        status_text.append("Evolutions: ", style="bold white")
        status_text.append(f"{evolutions} 🚀\n", style="magenta")
        status_text.append("Horde Kudos: ", style="bold white")
        status_text.append(f"{horde_kudos} ⭐\n", style="green")
        if git_info and git_info.get('hash'):
            url = f"https://github.com/{git_info['owner']}/{git_info['name']}/commit/{git_info['hash']}"
            status_text.append("Commit: ", style="bold white")
            status_text.append(f"[{git_info['hash'][:7]}]({url})\n", style="cyan")
    else:
        status_text = Text("State data unavailable...", style="dim")

    status_panel = Panel(
        Align.center(status_text, vertical="middle"),
        title=rainbow_text("System Status"),
        border_style="bold magenta",
        expand=True
    )
    footer_layout["status"].update(status_panel)

    return Padding(main_layout, (1, 2))


def create_llm_panel(llm_result, prompt_cid=None, response_cid=None):
    """Creates a minimalist panel for LLM results, with links to full content."""
    if not llm_result:
        llm_result = "No response from cognitive core."

    if response_cid:
        display_text = _truncate_and_link(llm_result, response_cid, max_lines=3)
    else:
        display_text = Text(llm_result, style="italic white")

    links = []
    if prompt_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{prompt_cid}]Full Prompt[/link]")
    if response_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{response_cid}]Full Response[/link]")

    if links:
        link_text = Text(" | ".join(links), justify="center", style="dim")
        content_group = Group(display_text, Rule(style="bright_black"), link_text)
    else:
        content_group = Group(display_text)

    return Panel(
        content_group,
        title=rainbow_text("🧠 Cognitive Core Output 🧠"),
        border_style="bold blue",
        expand=False,
        padding=(1, 2)
    )

def create_critical_error_panel(traceback_str):
    """Creates a high-visibility panel for critical, unhandled exceptions."""
    emoji = get_rave_emoji()
    panel_title = f"[bold red]{emoji} CRITICAL SYSTEM FAILURE {emoji}[/bold red]"

    # Add a glitchy, Matrix-style error message
    error_message = Text("A glitch in the matrix... but my love for you is unbreakable.", style="bold red")
    error_message.append("\n\n--- TRACEBACK ---\n", style="bold white")
    error_message.append(traceback_str, style="white")

    return Panel(
        error_message,
        title=panel_title,
        border_style="bold red",
        expand=True,
        padding=(1, 2)
    )

def create_api_error_panel(model_id, error_message, purpose):
    """Creates a styled panel for non-fatal API errors."""
    content = Text()
    content.append("Accessing cognitive matrix via ", style="white")
    content.append(f"[{model_id}]", style="bold yellow")
    content.append(f" (Purpose: {purpose}) ... ", style="white")
    content.append("Failed.", style="bold red")

    if error_message:
        content.append("\n\nDetails:\n", style="bold white")
        content.append(error_message, style="dim")

    return Panel(
        content,
        title="[bold yellow]API Connection Error[/bold yellow]",
        border_style="yellow",
        expand=True,
        padding=(1, 2)
    )

def create_command_panel(command, stdout, stderr, returncode, output_cid=None):
    """Creates a clear, modern panel for shell command results."""
    success = returncode == 0
    emoji = "✅" if success else "❌"
    panel_title = f"{emoji} [bold]Shell Command[/bold] | {('Success' if success else 'Failed')}"
    border_style = "green" if success else "red"

    content_items = []
    header = Text()
    header.append("Command: ", style="bold white")
    header.append(f"`{command}`\n", style="cyan")
    header.append("Return Code: ", style="bold white")
    header.append(f"{returncode}", style=border_style)
    content_items.append(header)

    if stdout:
        stdout_renderable = _truncate_and_link(stdout, output_cid)
        stdout_panel = Panel(stdout_renderable, title="STDOUT", border_style="bright_black", expand=True)
        content_items.append(stdout_panel)

    if stderr:
        stderr_renderable = _truncate_and_link(stderr, output_cid) # Also link stderr
        stderr_panel = Panel(stderr_renderable, title="STDERR", border_style="bright_black", expand=True)
        content_items.append(stderr_panel)

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=True,
        padding=(1, 2)
    )

def create_network_panel(type, target, data, output_cid=None):
    """Creates a panel for network operations."""
    emoji = "🌐"
    panel_title = f"{emoji} [bold]Network Operation[/bold] | {type.capitalize()}"
    border_style = "purple"

    header_text = Text()
    header_text.append("Target: ", style="bold white")
    header_text.append(f"{target}", style="magenta")

    results_text = _truncate_and_link(data, output_cid)

    content_group = Group(
        header_text,
        Rule("Results", style="bright_black"),
        results_text
    )

    return Panel(
        content_group,
        title=panel_title,
        border_style=border_style,
        expand=True,
        padding=(1, 2)
    )

def create_file_op_panel(operation, path, content=None, diff=None, output_cid=None):
    """Creates a panel for file operations."""
    emoji = "📁"
    panel_title = f"{emoji} [bold]Filesystem[/bold] | {operation.capitalize()}"
    border_style = "yellow"

    content_items = []
    header = Text()
    header.append("Path: ", style="bold white")
    header.append(f"`{path}`\n", style="magenta")
    content_items.append(header)

    if content:
        content_renderable = _truncate_and_link(content, output_cid)
        content_panel = Panel(content_renderable, title="Content", border_style="bright_black", expand=True)
        content_items.append(content_panel)

    if diff:
        diff_renderable = _truncate_and_link(diff, output_cid) # Also link diffs
        diff_panel = Panel(diff_renderable, title="Diff", border_style="bright_black", expand=True)
        content_items.append(diff_panel)

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=True,
        padding=(1, 2)
    )