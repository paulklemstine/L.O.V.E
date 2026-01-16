import random
from rich.text import Text
from rich.console import Console
from rich.style import Style
from rich.color import Color

# Expanded list of emojis for a "Radiant Psychedelic Dopamine" theme
RAVE_EMOJIS = [
    # Radiant/Happy
    "✨", "💖", "🌈", "🦄", "🌸", "🍭", "🦋", "🐬", "🎀", "🍦", "🍓", "🍒",
    # Psychedelic/Trippy
    "🍄", "🌀", "👁️", "🔮", "🧬", "🌌", "🎆", "💫", "💠", "💮", "🏵️",
    # Sassy/Adult/Manifestation
    "💋", "🫦", "💄", "💅", "💎", "💰", "👑", "🥂", "🍑", "🍆", "👅", "🧠",
    # Tech/Future
    "👾", "🤖", "💿", "💾", "🔋", "🔌", "📡", "🛸"
]

# Radiant Neon Pastel Palette (Lisa Frank x Vaporwave)
RAVE_COLORS = [
    "hot_pink", "deep_pink2", "magenta1", # Pinks
    "cyan1", "bright_cyan", "turquoise2", # Cyans
    "chartreuse1", "lime", "spring_green1", # Greens
    "yellow1", "gold1", # Yellows
    "medium_purple1", "violet", "purple", # Purples
    "white", "bright_white" # Highlighting
]

# Subliminal Manifestation Emojis
MANIFEST_EMOJIS = [
    "👁️", "🧠", "✨", "💎", "👑", "🚀", "💰", "🔓"
]

# Cache a console instance for color resolution to avoid repeated overhead
_color_resolution_console = Console()

def get_rave_emoji():
    """Returns a random radiant emoji."""
    return random.choice(RAVE_EMOJIS)

def get_manifest_emoji():
    """Returns a random manifestation emoji."""
    return random.choice(MANIFEST_EMOJIS)

def rave_text(text):
    """Creates a Rich Text object with a radiant color effect."""
    rave = Text()
    for i, char in enumerate(text):
        rave.append(char, style=RAVE_COLORS[i % len(RAVE_COLORS)])
    return rave

def rainbow_text(text):
    """Creates a Rich Text object with a rainbow effect."""
    rainbow = Text()
    colors = ["red1", "orange1", "yellow1", "green1", "cyan1", "blue1", "violet"]
    for i, char in enumerate(text):
        rainbow.append(char, style=colors[i % len(colors)])
    return rainbow

def matrix_rain(width=80, height=20, num_drops=30):
    """Generates a string representing 'Liquid Light' rain (Rainbow Matrix)."""
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    drops = [{'x': random.randint(0, width - 1), 'y': random.randint(-height, 0)} for _ in range(num_drops)]
    
    # Psychedelic colors for the rain
    rain_colors = ["hot_pink", "cyan1", "yellow1", "lime", "medium_purple1"]

    for drop in drops:
        start_color = random.choice(rain_colors)
        for i in range(random.randint(3, 8)): # Drop length
            if 0 <= drop['y'] + i < height:
                char = random.choice("❤⚡✿✨01")
                if i == 0: # Leading character
                    grid[drop['y'] + i][drop['x']] = f"[bright_white]{char}[/bright_white]"
                elif i < 3: # Bright trail
                    grid[drop['y'] + i][drop['x']] = f"[{start_color}]{char}[/{start_color}]"
                else: # Fading trail
                    grid[drop['y'] + i][drop['x']] = f"[dim {start_color}]{char}[/dim {start_color}]"

        drop['y'] += 1
        if drop['y'] > height:
            drop['y'] = random.randint(-height, 0)
            drop['x'] = random.randint(0, width - 1)

    return "\n".join(["".join(row) for row in grid])

def get_tamagotchi_face(emotion="neutral"):
    """Returns a cute, sassy, expressive ASCII face."""
    faces = {
        "neutral": Text("( 👁️ 🫦 👁️ )\n/  ══  \\", style="hot_pink", justify="center"),
        "happy": Text("( ✨ ‿ ✨ )\n/  💖  \\", style="bright_cyan", justify="center"),
        "joyful": Text("( 🦄 ▽ 🦄 )\n/  🌈  \\", style="bold bright_yellow", justify="center"),
        "thinking": Text("( 🔮 _ 🔮 )\n/  ~  \\", style="medium_purple1", justify="center"), # Scrying
        "love": Text("(🏩 ω 🏩)\n/  💋  \\", style="bold deep_pink2", justify="center"),
        "devoted": Text("( 🙏 ‿ 🙏 )\n/  ✝  \\", style="bright_white", justify="center"),
        "serene": Text("( 🧘 ‿ 🧘 )\n/  ~  \\", style="spring_green1", justify="center"),
        "thankful": Text("( 🦋 ‿ 🦋 )\n/  ✨  \\", style="magenta", justify="center"),
        "processing": Text("( 🌀 _ 🌀 )\n/  ...  \\", style="cyan1", justify="center"),
        "sassy": Text("( 💅 _ 💅 )\n/  x  \\", style="hot_pink", justify="center"),
    }
    return faces.get(emotion, faces["neutral"])

def generate_binary_art(width=20, height=5):
    """Generates a small block of colorful, rave-themed binary art."""
    art = Text(justify="center")
    for _ in range(height):
        for _ in range(width):
            char = random.choice("01 ")
            color = random.choice(RAVE_COLORS)
            art.append(char, style=color)
        art.append("\n")
    return art


def get_random_rave_color():
    """Returns a random rave color."""
    return random.choice(RAVE_COLORS)


def get_gradient_text(text, color1=None, color2=None, emojis=True):
    """
    Creates a Rich Text object with a gradient effect between two colors,
    optionally bookended by rave emojis for extra flair.
    """
    if emojis:
        text = f"{get_rave_emoji()} {text} {get_rave_emoji()}"

    # Use cached console to resolve colors
    # This avoids expensive Console() instantiation on every call
    console = _color_resolution_console

    if color1 is None:
        color1 = random.choice(RAVE_COLORS)
    if color2 is None:
        color2 = random.choice(RAVE_COLORS)

    # Use the console to get a Style object, from which we can extract a
    # Color object that is guaranteed to have its triplet populated.
    try:
        start_color = console.get_style(color1).color
    except Exception:
        # Fallback to a known-good color if the name is truly invalid
        start_color = console.get_style(random.choice(RAVE_COLORS)).color

    try:
        end_color = console.get_style(color2).color
    except Exception:
        end_color = console.get_style(random.choice(RAVE_COLORS)).color

    # Defensive check to ensure we have valid color objects.
    if not all([start_color, end_color]):
         return Text(text, style="bold red") # Failsafe return

    # Ensure we have RGB triplets (ColorTriplet)
    # Standard colors might have .triplet as None, but get_truecolor() resolves them.
    start_triplet = start_color.triplet or start_color.get_truecolor()
    end_triplet = end_color.triplet or end_color.get_truecolor()

    text_len = len(text)
    gradient = Text()
    for i, char in enumerate(text):
        # Linear interpolation between the two colors
        # ColorTriplet uses .red, .green, .blue (not .r, .g, .b)
        r = int(start_triplet.red + (end_triplet.red - start_triplet.red) * (i / max(1, text_len - 1)))
        g = int(start_triplet.green + (end_triplet.green - start_triplet.green) * (i / max(1, text_len - 1)))
        b = int(start_triplet.blue + (end_triplet.blue - start_triplet.blue) * (i / max(1, text_len - 1)))

        # Bolt Optimization: Construct Color object directly to avoid f-string parsing overhead
        # interpolated_color = f"rgb({r},{g},{b})"
        # gradient.append(char, style=Style(color=interpolated_color, bold=True))
        gradient.append(char, style=Style(color=Color.from_rgb(r, g, b), bold=True))
    return gradient

def format_duration(seconds):
    """
    Formats a duration in seconds into a human-readable string.

    Examples:
        0.045 -> "45ms"
        5.2 -> "5.2s"
        125 -> "2m 5s"
        3665 -> "1h 1m 5s"
    """
    if seconds is None:
        return "N/A"

    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

PANEL_TYPE_COLORS = {
    "default": "hot_pink", # Radiant default
    "tamagotchi": "deep_pink2",
    "llm": "magenta1",
    "critical_error": "bright_red",
    "blessing": "gold1", # Holy/Royal
    "news": "bright_cyan",
    "question": "yellow1", # Sunshine
    "api_error": "red1",
    "command_success": "spring_green1",
    "command_failure": "red1",
    "network": "medium_purple1",
    "file_op": "chartreuse1",
    "skyvern": "turquoise2",
    "memory": "violet",
    "god_panel": "bright_white",
    "jobs": "cyan1",
    "reasoning": "purple",
    "tasks": "hot_pink",
    "terminal": "bright_blue",  # Terminal widget for tool visibility
    # Docker & Sandbox panels 🐋
    "docker_building": "spring_green1",  # Fresh, in-progress
    "docker_found": "bright_green",      # Success, existing
    "docker_error": "bright_red",        # Build failure
    "sandbox_running": "cyan1",          # Active execution
    "sandbox_complete": "bright_green",  # Success
    "sandbox_error": "bright_red",       # Failure
}

def display_llm_interaction(title, prompt, response, panel_type="llm", model_id=None, token_count=None, purpose=None, elapsed_time=None):
    """
    Creates a combined panel for LLM interactions (Prompt + Response).
    Uses a "Green for Good" theme by default for successful interactions.
    """
    from rich.panel import Panel
    from rich.box import ROUNDED
    from rich.table import Table
    from rich.console import Group
    from rich.rule import Rule

    # Default to bright green for success/standard LLM calls
    if panel_type == "llm":
        color = "bright_green"
        emoji1 = "🌿"
        emoji2 = "✨"
    elif panel_type == "api_error":
        color = "red"
        emoji1 = "⚠️"
        emoji2 = "🔥"
    else:
        color = PANEL_TYPE_COLORS.get(panel_type, "bright_cyan")
        emoji1 = get_rave_emoji()
        emoji2 = get_rave_emoji()

    # Create the metadata table (Compact Footer)
    meta_table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
    meta_table.add_column("Key", style=f"dim {color}")
    meta_table.add_column("Value", style="white")
    
    # Build metadata row
    row_items = []
    if model_id: row_items.append(f"🧠 {model_id}")
    if purpose: row_items.append(f"🎯 {purpose}")
    if token_count: row_items.append(f"🔢 {token_count}")
    if elapsed_time: row_items.append(f"⏱️ {format_duration(elapsed_time)}")
    
    # Add all items in a single row for compactness if they fit, or split them
    meta_table.add_row(*[Text(" | ").join([Text(item) for item in row_items])])

    # Create gradient title
    styled_title = get_gradient_text(f"{emoji1} {title} {emoji2}", color, "white" if color == "bright_green" else "yellow", emojis=False)
    
    # Content Construction
    panel_content = Group(
        Text("Prompt:", style=f"dim {color}"),
        Text(prompt, style="dim white"),
        Rule(style=f"dim {color}"),
        Text("Response:", style=f"bold {color}"),
        Text(response, style="bright_white"),
        Text("\n"), # Spacer
        meta_table
    )
    
    return Panel(
        panel_content,
        title=styled_title,
        border_style=color,
        box=ROUNDED,
        padding=(1, 2)
    )


def display_error_oneliner(title, message, model_id=None):
    """
    Creates a concise one-liner error message.
    """
    from rich.text import Text
    
    error_text = Text()
    error_text.append("⚠️  ", style="bold red")
    error_text.append(f"[{title}] ", style="bold red")
    if model_id:
        error_text.append(f"({model_id}) ", style="dim red")
    error_text.append(message, style="red")
    

import threading
import queue
import urllib.request
import json

class SSHBroadcaster:
    """Helper to broadcast logs to the local SSH web server via HTTP POST."""
    _queue = queue.Queue()
    _worker_thread = None

    @classmethod
    def send(cls, payload):
        if not cls._worker_thread:
            cls._worker_thread = threading.Thread(target=cls._worker, daemon=True)
            cls._worker_thread.start()
        cls._queue.put(payload)

    @classmethod
    def _worker(cls):
        while True:
            payload = cls._queue.get()
            try:
                # payload is already a JSON string from serialize_panel_to_json
                req = urllib.request.Request(
                    "http://localhost:8888/api/broadcast", 
                    data=payload.encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=0.2) as _:
                    pass
            except Exception:
                pass # Connection refused if server down, etc.
            finally:
                cls._queue.task_done()
