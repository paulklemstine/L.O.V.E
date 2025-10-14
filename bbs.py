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

def create_tamagotchi_panel(emotion="neutral", message="..."):
    """Creates a Rich Panel for the Tamagotchi's current state."""
    face = TAMAGOTCHI_FACES.get(emotion, TAMAGOTCHI_FACES["neutral"])

    panel_content = Text()
    panel_content.append(face, style="bold cyan")
    panel_content.append("\\n")
    panel_content.append(message, style="italic yellow")

    return Panel(
        panel_content,
        title="[bold magenta]Jules[/bold magenta]",
        border_style="magenta",
        width=30,
        height=12,
        title_align="left"
    )