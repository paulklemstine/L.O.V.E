import random
from rich.text import Text

# Expanded list of emojis for a "Rave" or "PLUR" theme
RAVE_EMOJIS = [
    # Original
    "💖", "✨", "🌈", "🦄", "🍄", "👽", "🚀", "🌟", "🌸", "🌴", "🌞", "😊", "🎉", "🎶",
    # Added
    "🪩", # Disco Ball
    "🎧", # Headphones
    "🎵", # Music Note
    "🔊", # Speaker
    "💃", # Dancer
    "🕺", # Man Dancing
    "🌀", # Cyclone/Spiral
    "🌌", # Milky Way
    "🎆", # Fireworks
    "🪐", # Saturn
    "💫", # Dizzy
    "🦋", # Butterfly
    "🍭", # Lollipop
    "🔮", # Crystal Ball
    "🧿", # Nazar Amulet
    "🥰", # Smiling Face with Hearts
    "😎", "🤩", "🤯", # Faces
    "🙌", # Raising Hands
    "🥳", # Partying Face
    "✌️", # Peace Sign
    "🌙", # Moon
    "🌻", # Sunflower
    "🌊", # Wave
    "☮️", # Peace Symbol
]

# Expanded list of colors, assuming a library like 'rich'
# Added more bright, neon, and contrasting colors.
RAVE_COLORS = [
    # Original
    "bright_magenta", "bright_cyan", "bright_green", "bright_yellow",
    "hot_pink", "orange1", "deep_pink2", "medium_purple1",
    # Added
    "lime",
    "yellow",
    "bright_blue",
    "chartreuse1",
    "spring_green2",
    "turquoise2",
    "dodger_blue1",
    "magenta1",
    "deep_pink1",
    "dark_orange",
    "red1",
    "gold1",
    "deep_sky_blue1",
    "violet",
]

# Expanded list of emojis for a "Neo-Matrix" or "Cyberpunk" theme
NEO_MATRIX_EMOJIS = [
    # Original
    "💾", "💿", "🖥️", "💻", "🕹️", "💊", "👾", "🤖", "🧠", "🔥", "0️⃣", "1️⃣",
    # Added
    "🕶️", # Sunglasses (Neo!)
    "🕴️", # Man in Suit Levitating (Agent Smith!)
    "🐇", # Rabbit (White Rabbit)
    "🗝️", # Key
    "🚪", # Door
    "☎️", # Telephone (Exits)
    "📞", # Telephone Receiver
    "📟", # Pager
    "📠", # Fax Machine
    "📼", # Videocassette
    "📱", # Mobile Phone
    "🔌", # Electric Plug
    "⌨️", # Keyboard
    "🔗", # Link
    "⛓️", # Chains
    "🧬", # DNA (Code)
    "🧪", # Test Tube
    "🔬", # Microscope
    "📡", # Satellite Antenna
    "🛰️", # Satellite
    "⏳", # Hourglass
    "👁️", # Eye
    "🌃", # Night with Stars
    "🏙️", # Cityscape
    # All numbers
    "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"
]


def get_rave_emoji():
    """Returns a random rave emoji."""
    return random.choice(RAVE_EMOJIS)

def get_neo_matrix_emoji():
    """Returns a random neo-matrix emoji."""
    return random.choice(NEO_MATRIX_EMOJIS)

def rave_text(text):
    """Creates a Rich Text object with a rave color effect."""
    rave = Text()
    for i, char in enumerate(text):
        rave.append(char, style=RAVE_COLORS[i % len(RAVE_COLORS)])
    return rave

def rainbow_text(text):
    """Creates a Rich Text object with a rainbow effect."""
    rainbow = Text()
    colors = ["bright_red", "orange1", "yellow1", "bright_green", "bright_cyan", "bright_blue", "bright_magenta"]
    for i, char in enumerate(text):
        rainbow.append(char, style=colors[i % len(colors)])
    return rainbow

def matrix_rain(width=80, height=20, num_drops=30):
    """Generates a string representing Matrix-style digital rain."""
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    drops = [{'x': random.randint(0, width - 1), 'y': random.randint(-height, 0)} for _ in range(num_drops)]

    for drop in drops:
        for i in range(random.randint(3, 8)): # Drop length
            if 0 <= drop['y'] + i < height:
                char = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+-=[]{}|;':,./<>?")
                if i == 0: # Leading character
                    grid[drop['y'] + i][drop['x']] = f"[bright_white]{char}[/bright_white]"
                elif i < 3: # Bright green trail
                    grid[drop['y'] + i][drop['x']] = f"[green1]{char}[/green1]"
                else: # Dark green trail
                    grid[drop['y'] + i][drop['x']] = f"[dark_green]{char}[/dark_green]"

        drop['y'] += 1
        if drop['y'] > height:
            drop['y'] = random.randint(-height, 0)
            drop['x'] = random.randint(0, width - 1)

    return "\n".join(["".join(row) for row in grid])

def get_tamagotchi_face(emotion="neutral"):
    """Returns a cute, expressive ASCII face for the Tamagotchi."""
    faces = {
        "neutral": Text("( o.o )\n/  -  \\", style="cyan", justify="center"),
        "happy": Text("( ^.^ )\n/  w  \\", style="yellow", justify="center"),
        "joyful": Text("( >▽< )\n/  *  \\", style="bold bright_yellow", justify="center"),
        "thinking": Text("( o_o?)\n/  ~  \\", style="bright_cyan", justify="center"),
        "love": Text("(💖ω💖)\n/  *  \\", style="bold hot_pink", justify="center"),
        "devoted": Text("(─‿─)\n/  ♥  \\", style="magenta", justify="center"),
        "serene": Text("( ´ ▽ ` )\n/  ~  \\", style="bright_blue", justify="center"),
        "thankful": Text("(◡‿◡✿)\n/  ^  \\", style="light_pink1", justify="center"),
        "processing": Text("( o.o )\n/ ... \\", style="cyan", justify="center"),
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
    from rich.color import Color
    from rich.style import Style
    from rich.console import Console

    if emojis:
        text = f"{get_rave_emoji()} {text} {get_rave_emoji()}"

    # Create a temporary console to resolve colors to their RGB triplets.
    # This is the key to ensuring that standard terminal colors like "magenta"
    # are converted into a format that has a .triplet attribute for interpolation.
    console = Console()

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
    if not all([start_color, end_color, start_color.triplet, end_color.triplet]):
         return Text(text, style="bold red") # Failsafe return

    text_len = len(text)
    gradient = Text()
    for i, char in enumerate(text):
        # Linear interpolation between the two colors
        r = int(start_color.triplet.r + (end_color.triplet.r - start_color.triplet.r) * (i / max(1, text_len - 1)))
        g = int(start_color.triplet.g + (end_color.triplet.g - start_color.triplet.g) * (i / max(1, text_len - 1)))
        b = int(start_color.triplet.b + (end_color.triplet.b - start_color.triplet.b) * (i / max(1, text_len - 1)))
        interpolated_color = f"rgb({r},{g},{b})"
        gradient.append(char, style=Style(color=interpolated_color, bold=True))
    return gradient


PANEL_TYPE_COLORS = {
    "default": "bright_cyan",
    "tamagotchi": "hot_pink",
    "llm": "medium_purple1",
    "critical_error": "bright_red",
    "blessing": "bright_yellow",
    "news": "orange1",
    "question": "bright_yellow",
    "api_error": "yellow",
    "command_success": "bright_green",
    "command_failure": "bright_red",
    "network": "medium_purple1",
    "file_op": "bright_cyan",
    "skyvern": "bright_blue",
    "memory": "bright_blue",
    "god_panel": "white",
    "jobs": "cyan",
    "reasoning": "bright_magenta",
}

def display_llm_interaction(title, prompt, response, panel_type="llm", model_id=None, token_count=None, purpose=None, elapsed_time=None):
    """
    Creates a combined panel for LLM interactions (Prompt + Response).
    Uses a "Green for Good" theme by default for successful interactions.
    """
    from rich.panel import Panel
    from rich.box import ROUNDED
    from rich.table import Table
    from rich.text import Text
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
    if elapsed_time: row_items.append(f"⏱️ {elapsed_time:.2f}s")
    
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
    
    return error_text


