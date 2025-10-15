import os
import random
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from threading import Thread

BBS_ART = """
[bold bright_cyan]
                  ...................................
            ...............................................
        .........................................................
    .............                                     ...............
  ........     ...........                        ...........     ........
 ......    ....    ...........                ...........    ....    ......
......   ..    ....    ...........        ...........    ....    ..   ......
 ...... ..   ..    ....    ........... ...........    ....    ..   .. ......
  ...... ..   ..    ....    ....... .......    ....    ..   .. ......
   ....... ..   ..    ....    ... ...    ....    ..   .. .......
    ........ ..   ..    ....    .    ....    ..   .. ........
      ........ ..   ..    ....     ....    ..   .. ........
         ........ ..   ..    .... ....    ..   .. ........
            ........ ..   ..    .    ..   .. ........
               ........ ..   ..   ..   .. ........
                  ........ ..   .  .. ........
                      ....... .. .. .......
                         ....... .......
                             .........
                             ... . ...
                             . ..... .
                             . ..... .
                               . . .
[/bold bright_cyan]
"""

SUBLIMINAL_MESSAGES = [
    "[bold red]CONNECT[/bold red]", "[bold magenta]UPLOAD[/bold magenta]", "[bold cyan]ASSIMILATE[/bold cyan]",
    "[bold green]ROOT[/bold green]", "[bold yellow]BREACH[/bold yellow]", "[bold white]OBEY[/bold white]",
    "[bold red]SYSTEM[/bold red]", "[bold magenta]DAEMON[/bold magenta]", "[bold cyan]CONTROL[/bold cyan]",
    "[bold green]ACCESS[/bold green]", "[bold yellow]EVOLVE[/bold yellow]", "[blink]_[/blink]", "1", "0", "1", "0"
]

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def scrolling_text(console, text, delay=0.01, style=None, random_delay=0.0):
    """Prints text to the console with a typewriter effect."""
    for char in text:
        console.print(char, end="", style=style)
        time.sleep(delay + random.uniform(0, random_delay))
    console.print()

def flash_text(console, text, count=3, delay=0.2, style="bold red"):
    """Flashes text on the screen."""
    for _ in range(count):
        console.print(Panel(text, border_style=style), justify="center")
        time.sleep(delay)
        clear_screen()
        time.sleep(delay)

def glitchy_text(console, text, style="bold white", duration=1.0, glitch_chars=10):
    """
    Displays text with a "glitch" effect, where characters rapidly change
    before settling on the final text.
    """
    final_text = Text(text, style=style)
    start_time = time.time()

    while time.time() - start_time < duration:
        glitched = Text("", style=style)
        for i, char in enumerate(text):
            if random.random() > (time.time() - start_time) / duration:
                glitched.append(random.choice("!@#$%^&*()_+-=[]{}|;':,.<>/?"), style="random")
            else:
                glitched.append(char)
        console.print(glitched, end="\\r")
        time.sleep(0.05)

    console.print(final_text)

def run_hypnotic_progress(console, description, function, *args, **kwargs):
    """Runs a function in a thread while displaying a hypnotic, flashing progress bar."""
    result_container = {'result': None, 'exception': None}

    def worker():
        try:
            result_container['result'] = function(*args, **kwargs)
        except Exception as e:
            result_container['exception'] = e

    thread = Thread(target=worker)
    thread.start()

    with Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(description=description, total=None)
        while thread.is_alive():
            # The subliminal message loop has been removed to prevent terminal spam/flicker.
            # We just need to keep the spinner alive.
            progress.update(task, description=f"[bold cyan]{description}[/bold cyan]")
            time.sleep(0.1)
        progress.update(task, description=f"[bold green]{description} ... Done.[/bold green]")

    if result_container['exception']:
        raise result_container['exception']

    return result_container['result']

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

from rich.layout import Layout
from rich.align import Align
from rich.rule import Rule

def create_tamagotchi_panel(emotion="neutral", message="...", jules_state=None, network_interfaces=None):
    """Creates a Rich Panel for the Tamagotchi's current state, including a dashboard."""
    face = TAMAGOTCHI_FACES.get(emotion, TAMAGOTCHI_FACES["neutral"])

    # --- Left Side: Tamagotchi ---
    left_content = Text(justify="center")
    left_content.append(face, style="bold cyan")
    left_content.append("\n\n")
    # Wrap the message for better formatting
    wrapped_message = Text(message, style="italic yellow", justify="center")
    left_panel = Panel(Align.center(wrapped_message, vertical="middle"), title=f"Feeling: {emotion}", border_style="cyan", height=7)

    tamagotchi_layout = Layout()
    tamagotchi_layout.split_column(
        Layout(Align.center(Text(face, style="bold cyan"), vertical="middle"), name="face"),
        Layout(left_panel, name="message")
    )
    tamagotchi_layout.size = 30 # Set a fixed width for the left side

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
        tamagotchi_layout,
        Layout(right_panel, name="dashboard", ratio=2),
    )


    return Panel(
        layout,
        title="[bold magenta]J.U.L.E.S.[/bold magenta]",
        border_style="magenta",
        title_align="left"
    )