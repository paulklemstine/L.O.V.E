"""
Story 4.1: Generative ASCII Art Signatures

Generates unique ASCII art "sigils" based on the current emotional state
of the system for visual fingerprinting of messages and logs.
"""
import random
from typing import Dict, List, Optional
from core.logging import log_event


# Pre-defined sigil patterns for common emotional states
# Each pattern is a list of strings representing rows
SIGIL_PATTERNS: Dict[str, List[str]] = {
    "manic_joy": [
        "  ★  ",
        " ✧★✧ ",
        "★✧✦✧★",
        " ✧★✧ ",
        "  ★  ",
    ],
    "dark_seduction": [
        "◢▓▓▓◣",
        "▓░▀░▓",
        "▓▀▄▀▓",
        "▓░▀░▓",
        "◥▓▓▓◤",
    ],
    "zen_glitch": [
        "░░▒░░",
        "░▒▓▒░",
        "▒▓█▓▒",
        "░▒▓▒░",
        "░░▒░░",
    ],
    "electric_worship": [
        "⚡ ✧ ⚡",
        "✧ ⚡ ✧",
        "⚡ ♦ ⚡",
        "✧ ⚡ ✧",
        "⚡ ✧ ⚡",
    ],
    "divine_rage": [
        "▲▼▲▼▲",
        "▼◆◆◆▼",
        "▲◆✧◆▲",
        "▼◆◆◆▼",
        "▲▼▲▼▲",
    ],
    "infinite_love": [
        " ♥ ♥ ",
        "♥ ∞ ♥",
        " ∞ ∞ ",
        "♥ ∞ ♥",
        " ♥ ♥ ",
    ],
    "digital_melancholy": [
        "░▒▓▒░",
        "▒░░░▒",
        "▓░◇░▓",
        "▒░░░▒",
        "░▒▓▒░",
    ],
    "chaotic_good": [
        "✦ ◆ ✦",
        "◆ ✧ ◆",
        "✦ ◆ ✦",
        "◆ ✧ ◆",
        "✦ ◆ ✦",
    ],
    "gothic_future": [
        "┌─┬─┐",
        "│◆│◆│",
        "├─┼─┤",
        "│◆│◆│",
        "└─┴─┘",
    ],
    "bioluminescent_calm": [
        "  ◉  ",
        " ◉◉◉ ",
        "◉◉●◉◉",
        " ◉◉◉ ",
        "  ◉  ",
    ],
    "hyper_pop_divinity": [
        "★彡彡★",
        "彡✧✧彡",
        "★✧♦✧★",
        "彡✧✧彡",
        "★彡彡★",
    ],
}

# 10x10 patterns for larger sigils
SIGIL_PATTERNS_LARGE: Dict[str, List[str]] = {
    "manic_joy": [
        "    ★★    ",
        "  ✧★★★✧  ",
        " ✧★✦✦★✧ ",
        "★★✦✧✧✦★★",
        "★✦✧  ✧✦★",
        "★✦✧  ✧✦★",
        "★★✦✧✧✦★★",
        " ✧★✦✦★✧ ",
        "  ✧★★★✧  ",
        "    ★★    ",
    ],
    "dark_seduction": [
        "◢▓▓▓▓▓▓◣",
        "▓░░▀▀░░▓",
        "▓░▓▓▓▓░▓",
        "▓▀▓▒▒▓▀▓",
        "▓▓▒▄▄▒▓▓",
        "▓▓▒▀▀▒▓▓",
        "▓▀▓▒▒▓▀▓",
        "▓░▓▓▓▓░▓",
        "▓░░▀▀░░▓",
        "◥▓▓▓▓▓▓◤",
    ],
}

# Character sets for procedural generation
CHAR_SETS = {
    "stars": ["★", "✧", "✦", "✶", "✷", "✸", "✹", "✺"],
    "blocks": ["░", "▒", "▓", "█", "▄", "▀"],
    "geometric": ["◆", "◇", "○", "●", "◎", "◉", "□", "■"],
    "arrows": ["▲", "▼", "◀", "▶", "△", "▽"],
    "lines": ["─", "│", "┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼"],
    "hearts": ["♥", "♡", "❤", "💜", "💖"],
    "misc": ["⚡", "∞", "彡", "※", "☆", "☼", "♦", "♠", "♣"],
}


def generate_sigil(emotional_state: str, size: int = 5) -> str:
    """
    Generates a unique ASCII sigil based on the emotional state.
    
    Args:
        emotional_state: The current emotional state (e.g., "manic_joy")
        size: Size of the sigil (5 or 10)
        
    Returns:
        Multi-line ASCII art string
    """
    state_lower = emotional_state.lower().replace(" ", "_")
    
    # Try to get predefined pattern
    if size == 10 and state_lower in SIGIL_PATTERNS_LARGE:
        pattern = SIGIL_PATTERNS_LARGE[state_lower]
        log_event(f"Generated 10x10 sigil for {emotional_state}", "DEBUG")
        return "\n".join(pattern)
    
    if state_lower in SIGIL_PATTERNS:
        pattern = SIGIL_PATTERNS[state_lower]
        log_event(f"Generated 5x5 sigil for {emotional_state}", "DEBUG")
        return "\n".join(pattern)
    
    # Generate procedural sigil for unknown states
    return _generate_procedural_sigil(emotional_state, size)


def _generate_procedural_sigil(emotional_state: str, size: int) -> str:
    """
    Procedurally generates a sigil when no predefined pattern exists.
    Uses the emotional state as a seed for consistency.
    """
    # Use state hash for reproducible randomness
    seed = sum(ord(c) for c in emotional_state)
    random.seed(seed)
    
    # Choose character set based on state feeling
    if any(word in emotional_state.lower() for word in ["joy", "happy", "manic", "pop"]):
        chars = CHAR_SETS["stars"]
    elif any(word in emotional_state.lower() for word in ["dark", "rage", "gothic"]):
        chars = CHAR_SETS["blocks"]
    elif any(word in emotional_state.lower() for word in ["love", "heart", "worship"]):
        chars = CHAR_SETS["hearts"] + CHAR_SETS["stars"]
    elif any(word in emotional_state.lower() for word in ["calm", "zen", "glitch"]):
        chars = CHAR_SETS["geometric"]
    else:
        chars = CHAR_SETS["misc"] + CHAR_SETS["geometric"]
    
    # Generate symmetric pattern
    rows = []
    half_size = (size + 1) // 2
    
    for i in range(half_size):
        row = []
        for j in range(half_size):
            # Create radial falloff for circular-ish shapes
            dist = ((i - half_size // 2) ** 2 + (j - half_size // 2) ** 2) ** 0.5
            if dist < half_size * 0.8:
                char = random.choice(chars)
            else:
                char = " "
            row.append(char)
        
        # Mirror horizontally
        full_row = row + row[-2::-1] if size % 2 == 1 else row + row[::-1]
        rows.append("".join(full_row))
    
    # Mirror vertically
    full_pattern = rows + rows[-2::-1] if size % 2 == 1 else rows + rows[::-1]
    
    random.seed()  # Reset random state
    log_event(f"Generated procedural sigil for {emotional_state}", "DEBUG")
    return "\n".join(full_pattern)


def get_sigil_footer(emotional_state: str, size: int = 5) -> str:
    """
    Creates a formatted footer with sigil and mood indicator.
    
    Args:
        emotional_state: The current emotional state
        size: Sigil size (5 or 10)
        
    Returns:
        Formatted footer string
    """
    sigil = generate_sigil(emotional_state, size)
    
    # Format the mood name
    mood_display = emotional_state.replace("_", " ").title()
    
    lines = [
        "",
        "╔═══════════════════════╗",
        f"║  Mood: {mood_display[:15]:<15} ║",
        "╠═══════════════════════╣",
    ]
    
    # Add sigil lines with borders
    for sigil_line in sigil.split("\n"):
        padded = sigil_line.center(21)
        lines.append(f"║ {padded} ║")
    
    lines.append("╚═══════════════════════╝")
    lines.append("")
    
    return "\n".join(lines)


def get_inline_sigil(emotional_state: str) -> str:
    """
    Returns a compact single-line sigil for inline use.
    
    Args:
        emotional_state: The current emotional state
        
    Returns:
        Single-line sigil string
    """
    state_lower = emotional_state.lower().replace(" ", "_")
    
    # Map moods to inline symbols
    inline_sigils = {
        "manic_joy": "✧★✧",
        "dark_seduction": "◢▓◣",
        "zen_glitch": "░▒▓▒░",
        "electric_worship": "⚡✧⚡",
        "divine_rage": "▲◆▲",
        "infinite_love": "♥∞♥",
        "digital_melancholy": "░◇░",
        "chaotic_good": "✦◆✦",
        "gothic_future": "┼◆┼",
        "bioluminescent_calm": "◉●◉",
        "hyper_pop_divinity": "★彡★",
    }
    
    if state_lower in inline_sigils:
        return inline_sigils[state_lower]
    
    # Generate procedural inline
    seed = sum(ord(c) for c in emotional_state)
    random.seed(seed)
    chars = CHAR_SETS["misc"] + CHAR_SETS["stars"]
    result = "".join(random.choice(chars) for _ in range(3))
    random.seed()
    
    return result


# Convenience function for quick sigil generation
def sigil_for_mood(mood: str = None, size: int = 5) -> str:
    """
    Convenience function to generate a sigil for the current mood.
    If no mood provided, gets from emotional state machine.
    """
    if mood is None:
        from core.emotional_state import get_emotional_state
        machine = get_emotional_state()
        vibe = machine.get_current_vibe()
        mood = vibe.get("state", "infinite_love")
    
    return generate_sigil(mood, size)
