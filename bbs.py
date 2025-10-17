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

def run_hypnotic_progress(console, description, function, *args, **kwargs):
    """
    Runs a function and displays a status message.
    Can be silenced for non-critical background tasks.
    """
    silent = kwargs.pop('silent', False)

    if not silent:
        console.print(f"[bold cyan]{description}[/bold cyan]")

    try:
        result = function(*args, **kwargs)
        if not silent:
            console.print(f"[bold green]{description} ... Done.[/bold green]")
        return result
    except Exception as e:
        if not silent:
            console.print(f"[bold red]{description} ... Failed.[/bold red]")
        raise e

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