import os
import random
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from threading import Thread

BBS_ART = """
[bold bright_green]
              .o+o.
             / \\`o'
            /   \\
           /     \\
      +--./  !!!  \\.--+
      |  ;  (o_o)  ;  |
      |  '. \\_-_/ ,'  |
      |    '-...-'    |
      |      \\_/      |
      +--..__ | __..--+
            .' '.
           /     \\
          /       \\
         /         \\
[bold white]    E . V . I . L .    O N L I N E[/bold white][/bold bright_green]
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