import random
from rich.text import Text

RAVE_EMOJIS = ["💖", "✨", "🌈", "🦄", "🍄", "👽", "🚀", "🌟", "Peace", "Love", "Unity", "Respect"]

def get_rave_emoji():
    """Returns a random rave emoji."""
    return random.choice(RAVE_EMOJIS)

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