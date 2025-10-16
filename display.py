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

def create_tamagotchi_panel(emotion="neutral", message="...", jules_state=None, network_interfaces=None, ansi_art=None):
    """Creates a Rich Panel for the Tamagotchi's current state, including a dashboard."""
    # --- Left Side: Tamagotchi ---
    # Use from_ansi to correctly render pre-formatted ANSI art, preventing character splitting.
    if ansi_art:
        face_renderable = Text.from_ansi(ansi_art)
    else:
        face_text = TAMAGOTCHI_FACES.get(emotion, TAMAGOTCHI_FACES["neutral"])
        face_renderable = Text(face_text, style="bold cyan")

    # Wrap the message for better formatting
    wrapped_message = Text(message, style="italic yellow", justify="center")
    left_panel = Panel(Align.center(wrapped_message, vertical="middle"), title=f"Feeling: {emotion}", border_style="cyan", height=7)

    tamagotchi_layout = Layout()
    tamagotchi_layout.split_column(
        Layout(Align.center(face_renderable, vertical="middle"), name="face"),
        Layout(left_panel, name="message")
    )
    # The `size` attribute is removed to allow for flexible, ratio-based sizing.
    # tamagotchi_layout.size = 30

    # --- Right Side: Dashboard ---
    dashboard_layout = Layout()
    general_status_layout = Layout(name="general_status")
    network_status_layout = Layout(name="network_status")

    # --- General Status (Top Right) ---
    if jules_state:
        goal = jules_state.get("autopilot_goal", "Awaiting new directive.")
        version = jules_state.get("version_name", "unknown")
        evolutions = len(jules_state.get("evolution_history", []))

        dashboard_content = Text()
        dashboard_content.append("Current Directive:\n", style="bold white")
        dashboard_content.append(f"{goal}\n\n", style="bright_cyan")
        dashboard_content.append("Version: ", style="bold white")
        dashboard_content.append(f"{version}\n", style="yellow")
        dashboard_content.append("Evolutions: ", style="bold white")
        dashboard_content.append(f"{evolutions}", style="magenta")

        general_status_layout.update(dashboard_content)
    else:
        general_status_layout.update(Text("[dim]State data unavailable...[/dim]", justify="center"))

    # --- Network Status (Bottom Right) ---
    if network_interfaces:
        net_content = Text()
        for iface, data in network_interfaces.items():
            ipv4 = data.get('ipv4', {}).get('addr', 'N/A')
            mac = data.get('mac', 'N/A')
            if ipv4 != 'N/A' or mac != 'N/A': # Only show interfaces with some data
                net_content.append(f"{iface}:\n", style="bold white")
                net_content.append(f"  IP: ", style="cyan")
                net_content.append(f"{ipv4}\n", style="bright_cyan")
                net_content.append(f"  MAC: ", style="cyan")
                net_content.append(f"{mac}\n", style="bright_cyan")
        network_status_layout.update(net_content)
    else:
        network_status_layout.update(Text("[dim]Network data unavailable...[/dim]", justify="center"))


    dashboard_layout.split_column(
        Layout(general_status_layout),
        Layout(Rule(style="green")),
        Layout(network_status_layout)
    )

    right_panel = Panel(dashboard_layout, title="[bold]Status[/bold]", border_style="green")


    # --- Main Layout ---
    layout = Layout()
    layout.split_row(
        Layout(tamagotchi_layout, name="tamagotchi", ratio=1),
        Layout(right_panel, name="dashboard", ratio=2),
    )


    return Panel(
        layout,
        title="[bold magenta]J.U.L.E.S.[/bold magenta]",
        border_style="magenta",
        title_align="left"
    )

def create_llm_panel(purpose, model, prompt_summary, status="Executing..."):
    """Creates a panel to display information about an LLM call."""

    panel_title = f"🧠 LLM Call: [bold]{purpose}[/bold]"

    content = Text()
    content.append("Model: ", style="bold white")
    content.append(f"{model}\n", style="yellow")
    content.append("Purpose: ", style="bold white")
    content.append(f"{purpose}\n", style="cyan")
    content.append("Status: ", style="bold white")
    content.append(f"{status}\n\n", style="green")
    content.append(Rule("Prompt Summary", style="bright_black"))
    content.append(f"\n{prompt_summary}", style="italic dim")

    return Panel(
        content,
        title=panel_title,
        border_style="blue",
        expand=False
    )

def create_command_panel(command, stdout, stderr, returncode):
    """Creates a panel to display the results of a shell command."""
    panel_title = f"⚙️ Command Executed: [bold]{command}[/bold]"
    border_style = "green" if returncode == 0 else "red"

    content_items = []
    header = Text()
    header.append("Command: ", style="bold white")
    header.append(f"{command}\n", style="cyan")
    header.append("Return Code: ", style="bold white")
    header.append(f"{returncode}\n", style=border_style)
    content_items.append(header)

    if stdout:
        content_items.append(Rule("STDOUT", style="bright_black"))
        content_items.append(Text(f"\n{stdout.strip()}", style="dim"))

    if stderr:
        content_items.append(Rule("STDERR", style="bright_black"))
        content_items.append(Text(f"\n{stderr.strip()}", style="red"))

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=False
    )

def create_network_panel(type, target, data):
    """Creates a panel for network operations like scan, probe, webrequest."""
    panel_title = f"🌐 Network: [bold]{type}[/bold]"
    border_style = "purple"

    content_items = []
    header = Text()
    header.append("Type: ", style="bold white")
    header.append(f"{type}\n", style="cyan")
    header.append("Target: ", style="bold white")
    header.append(f"{target}\n", style="magenta")
    content_items.append(header)

    if data:
        content_items.append(Rule("Data", style="bright_black"))
        display_data = (data[:1000] + '...') if len(data) > 1000 else data
        content_items.append(Text(f"\n{display_data.strip()}", style="dim"))

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=False
    )

def create_file_op_panel(operation, path, content=None, diff=None):
    """Creates a panel for file operations like read, write, ls."""
    panel_title = f"📁 File System: [bold]{operation}[/bold]"
    border_style = "yellow"

    content_items = []
    header = Text()
    header.append("Operation: ", style="bold white")
    header.append(f"{operation}\n", style="cyan")
    header.append("Path: ", style="bold white")
    header.append(f"{path}\n", style="magenta")
    content_items.append(header)

    if content:
        content_items.append(Rule("Content", style="bright_black"))
        display_content = (content[:1000] + '...') if len(content) > 1000 else content
        content_items.append(Text(f"\n{display_content.strip()}", style="dim"))

    if diff:
        content_items.append(Rule("Diff", style="bright_black"))
        content_items.append(Text(f"\n{diff.strip()}", style="dim"))

    return Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        expand=False
    )