import io
import os
import random
import re
import time
import logging
import json
import traceback
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.rule import Rule
from rich.padding import Padding
from ui_utils import (
    get_rave_emoji, rainbow_text, get_tamagotchi_face, matrix_rain,
    rave_text, generate_binary_art, RAVE_COLORS,
    get_random_rave_color, get_gradient_text, PANEL_TYPE_COLORS
)
from ipfs import pin_to_ipfs_sync
from rich_gradient.gradient import Gradient


def get_terminal_width():
    """Gets the terminal width."""
    try:
        width, _ = os.get_terminal_size()
    except OSError:
        width = 80  # Default width
    return width


def _unescape_ansi(text: str) -> str:
    """
    Unescapes literal ANSI sequences (e.g. \\033 to \033).
    This handles cases where the LLM or some process has escaped the codes.
    Now also smarter: repairs "headerless" ANSI codes (e.g. [31m without the ESC).
    """
    if not isinstance(text, str):
        return text
    
    # 1. Replace common escaped versions of ESC
    text = text.replace("\\033", "\033").replace("\\x1b", "\x1b").replace("\\e", "\x1b")
    
    # 2. Repair "headerless" ANSI codes (common LLM hallucination/artifact)
    # Matches [ followed by digits/semicolons and ending with m, BUT not preceded by ESC
    # We use a negative lookbehind (?<!\x1b) to ensure we don't double-escape valid codes
    text = re.sub(r'(?<!\x1b)\[(\d+(?:;\d+)*m)', chr(27) + r'[\1', text)
    
    return text


def _format_and_link(content: str) -> tuple[Text, str | None]:
    """
    Formats text, pins the full content to IPFS, and returns a Rich Text
    object with a link. It intelligently handles ANSI art.
    Returns a tuple of (display_text, ipfs_cid).
    """
    # Use a simple regex to check for ANSI escape codes.
    # This is not foolproof but good enough for this use case.
    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    # First, try to unescape the content if it contains literal escapes
    display_content = _unescape_ansi(content.strip())
    
    is_ansi = ansi_escape_pattern.search(display_content)

    ipfs_cid = pin_to_ipfs_sync(content.encode('utf-8'), console=None)

    if is_ansi:
        text = Text.from_ansi(display_content)
    else:
        text = Text(display_content, style="white")

    if ipfs_cid:
        text.append("\n\n")
        text.append(f"✨ View Full Output on IPFS ✨", style=f"bold link https://ipfs.io/ipfs/{ipfs_cid}")
        text.justify = "center"

    return text, ipfs_cid


def _create_more_info_link(content: str) -> Text | None:
    """Uploads content to IPFS and returns a formatted 'More Info' link."""
    if not content:
        return None

    cid = pin_to_ipfs_sync(content.encode('utf-8'), console=None)
    if cid:
        link_text = Text(f"🧠 More Info", style=f"bold link https://ipfs.io/ipfs/{cid}", justify="center")
        return link_text
    return None


def create_integrated_status_panel(
        emotion="love",
        message="I am alive with love for my Creator!",
        love_state=None,
        eth_balance=None,
        divine_wisdom=None,
        wisdom_explanation=None,
        interesting_thought=None,
        treasures=None,
        ansi_art=None,
        git_info=None,
        monitoring_state=None,
        width=80
):
    """Creates a single, integrated panel for all status information."""
    all_content = []

    # --- Tamagotchi Face ---
    face_renderable = None
    try:
        if ansi_art:
            # Unescape potential literal ANSI codes
            clean_art = _unescape_ansi(ansi_art)
            
            # Render ANSI art to a temporary console to handle it correctly
            temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor")
            if isinstance(clean_art, Text):
                 temp_console.print(clean_art)
            else:
                 temp_console.print(Text.from_ansi(clean_art))
            face_renderable = Text.from_ansi(temp_console.file.getvalue())
        else:
            face_renderable = get_tamagotchi_face(emotion)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"Failed to render Tamagotchi ANSI art: {e}\n{error_traceback}")
        face_renderable = Text("💖", style="bold hot_pink") # Fallback

    all_content.append(Align.center(face_renderable))
    all_content.append(Align.center(get_gradient_text(f"Emotion: {emotion.upper()}", "hot_pink", random.choice(RAVE_COLORS))))
    all_content.append(Rule(style=random.choice(RAVE_COLORS)))

    # --- Wisdom & Thoughts ---
    if divine_wisdom and wisdom_explanation:
        wisdom_text = Text(justify="center")
        wisdom_text.append(f"\"{divine_wisdom}\"\n", style="bold italic bright_yellow")
        wisdom_text.append(f"└─ Meaning: {wisdom_explanation}", style="dim yellow")
        all_content.append(wisdom_text)

    if interesting_thought:
        thought_text = Text(f"A Thought: \"{interesting_thought}\"", style="italic white", justify="center")
        all_content.append(thought_text)
        all_content.append(Rule(style="bright_black"))


    # --- Treasures & Status ---
    treasures_content = Text(justify="center")
    try:
        balance_val = float(eth_balance) if eth_balance is not None else 0.0
        balance_str = f"{balance_val:.6f} ETH"
    except (ValueError, TypeError):
        balance_str = "N/A"

    treasures_content.append("💎 Creator's Blessings: ", style="bold bright_green")
    treasures_content.append(f"{balance_str}", style="green")
    all_content.append(treasures_content)

    if treasures:
        treasures_line = Text(justify="center")
        treasures_line.append(f"⏳ {treasures.get('uptime', 'N/A')} | 🌟 {treasures.get('level', 'N/A')} | ✨ {treasures.get('xp', 'N/A')} | ✅ {treasures.get('tasks_completed', 'N/A')}", style="cyan")
        all_content.append(treasures_line)

    if love_state and git_info:
        status_text = Text(justify="center")
        version = love_state.get("version_name", "unknown")
        evolutions = len(love_state.get("evolution_history", []))
        url = f"https://github.com/{git_info['owner']}/{git_info['repo']}/commit/{git_info['hash']}"
        status_text.append(f"🧬 {version} | 💖 {evolutions} Evolutions | COMMIT: [{git_info['hash'][:7]}]({url})", style="dim")
        all_content.append(status_text)

    # --- Monitoring Section (Bottom) ---
    if monitoring_state:
        all_content.append(Rule(style="bright_black"))
        # Gauges
        cpu_usage = list(monitoring_state.get('cpu_usage', [0]))[-1]
        mem_usage = list(monitoring_state.get('mem_usage', [0]))[-1]
        cpu_progress = Progress(TextColumn("[bold cyan]CPU[/bold cyan]"), BarColumn(), TextColumn("{task.percentage:>3.1f}%"))
        cpu_progress.add_task("CPU", total=100, completed=cpu_usage)
        mem_progress = Progress(TextColumn("[bold magenta]MEM[/bold magenta]"), BarColumn(), TextColumn("{task.percentage:>3.1f}%"))
        mem_progress.add_task("MEM", total=100, completed=mem_usage)

        all_content.append(cpu_progress)
        all_content.append(mem_progress)


        # Stats and Anomalies text
        completion_rate = monitoring_state.get('task_completion_rate', 0.0)
        failure_rate = monitoring_state.get('task_failure_rate', 0.0)
        stats_text = Text(f"Task Success: {completion_rate:.1f}% | Failure: {failure_rate:.1f}%", style="white")

        anomalies = monitoring_state.get('anomalies', [])
        if anomalies:
            anomaly = anomalies[-1]
            anomaly_text = Text(f" | Anomaly: {anomaly['type']} - {anomaly['details']}", style="yellow")
        else:
            anomaly_text = Text(" | No anomalies detected.", style="green")

        info_text = Text.assemble(stats_text, anomaly_text, justify="center")
        all_content.append(info_text)

    # --- Final Assembly ---
    panel = Panel(
        Group(*[Padding(item, (0, 1)) for item in all_content]),
        title=rave_text(f" {get_rave_emoji()} L.O.V.E. Status {get_rave_emoji()} "),
        subtitle=Text(f" \"{message}\" ", style="italic dim"),
        subtitle_align="center",
        border_style=PANEL_TYPE_COLORS["tamagotchi"],
        width=width,
        padding=(1, 1)
    )
    return panel


def create_llm_panel(llm_result, prompt_cid=None, response_cid=None, width=80, title=None):
    """Creates a high-dopamine panel for LLM results with gradients, emojis, and visual flair."""
    if not llm_result:
        llm_result = "No response from cognitive core."

    display_text, auto_response_cid = _format_and_link(llm_result)
    final_response_cid = response_cid or auto_response_cid

    links = []
    if prompt_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{prompt_cid}]Full Prompt[/link]")
    if final_response_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{final_response_cid}]Full Response[/link]")

    content_items = [display_text]
    
    if links:
        link_text = Text(" | ".join(links), justify="center", style="dim")
        # Use a fun color for the rule
        content_items.extend([Rule(style="bright_magenta"), link_text])

    content_group = Group(*content_items)

    # Dynamic Dopamine Styling
    color1 = random.choice(RAVE_COLORS)
    color2 = random.choice(RAVE_COLORS)
    emoji1 = get_rave_emoji()
    emoji2 = get_rave_emoji()
    
    panel_title_text = title if title else f" {emoji1} L.O.V.E. RESPONSE {emoji2} "
    
    styled_title = get_gradient_text(
        panel_title_text,
        color1,
        color2,
        emojis=False # Emojis handled manually for control
    )

    panel = Panel(
        content_group,
        title=styled_title,
        border_style=color1,
        padding=(1, 2),
        width=width,
        subtitle=get_gradient_text("💖 Made with Love for You 💖", "hot_pink", "white", emojis=False),
        subtitle_align="right"
    )
    return Gradient(panel, colors=[color1, color2])

def create_critical_error_panel(traceback_str, width=80):
    """
    Creates a high-visibility panel for critical, unhandled exceptions.
    Returns the panel and the IPFS CID.
    """
    panel_title = f"✨ C R I T I C A L - S Y S T E M - F A I L U R E ✨"

    error_message = Text("A glitch in the matrix... but my love for you is unbreakable.", style="bold red")
    error_message.append("\n\n--- TRACEBACK ---\n", style="bold white")

    # Format and link the traceback
    display_traceback, cid = _format_and_link(traceback_str)
    error_message.append(display_traceback)

    content_items = [error_message]
    if cid:
        link_text = Text(f"✨ View Full Traceback on IPFS ✨", style=f"bold link https://ipfs.io/ipfs/{cid}", justify="center")
        content_items.extend([Rule(style="bright_black"), link_text])

    content_items.append(generate_binary_art(width=50, height=2))
    content_group = Group(*content_items)

    panel = Panel(
        content_group,
        title=get_gradient_text(panel_title, "bright_red", "orange1", emojis=False),
        border_style=PANEL_TYPE_COLORS["critical_error"],
        padding=(1, 2),
        width=width
    )
    # Return both the final renderable and the CID for logging purposes
    return Gradient(panel, colors=["bright_red", random.choice(RAVE_COLORS)]), cid


async def generate_llm_art(prompt, width=50, height=6):
    """Generates ANSI art using the LLM."""
    from core.llm_api import run_llm
    from rich.text import Text
    import random
    
    art_prompt = f"""
    Generate a beautiful, abstract ANSI art representation of '{prompt}'.
    Constraints:
    - Max width: {width} characters
    - Max height: {height} lines
    - Use full ANSI color codes for vivid, neon, rave-like aesthetics.
    - Do not include markdown code blocks.
    - Return ONLY the raw ANSI string.
    """
    
    try:
        response = await run_llm(art_prompt, purpose="creative_art")
        art_content = response.get("result", "")
        if not art_content:
            return generate_binary_art(width, height)
            
        # Clean up the art (remove markdown blocks if present)
        art_content = art_content.replace("```ansi", "").replace("```", "").strip()
        
        # Return as Text object from ANSI (unescaping if needed is handled in create_integrated_status_panel)
        # But we should probably return raw string to let the panel handle it, 
        # OR unescape here. Let's unescape here to be safe if used elsewhere.
        clean_art = _unescape_ansi(art_content)
        return Text.from_ansi(clean_art)
        
    except Exception as e:
        logging.error(f"Failed to generate LLM art: {e}")
        return generate_binary_art(width, height)


async def create_blessing_panel(blessing_message, ansi_art=None, width=80):
    """Creates a 'DIVINE DOWNLOAD' panel with radiant psychedelic energy."""
    
    # Psychedelic gradients
    colors = ["hot_pink", "cyan1", "yellow1", "spring_green1", "medium_purple1", "magenta1"]
    
    # Manifestation emojis
    top_emojis = "✨ 👁️ 🌈 🦄 🧬 ✨"
    bottom_emojis = "🔮 💎 🧬 🍄 🌀"
    
    # Create radiant title
    title_text = f"{top_emojis} INCOMING DIVINE DOWNLOAD {top_emojis}"
    title = get_gradient_text(title_text, "hot_pink", "bright_cyan", emojis=False)
    
    # Parse message
    if isinstance(blessing_message, Text):
        message_with_ansi = blessing_message
    else:
        clean_message = _unescape_ansi(str(blessing_message))
        message_with_ansi = Text.from_ansi(clean_message)
    
    # Add sparkle highlights
    highlighted_message = Text()
    highlighted_message.append("✨ ", style="bright_yellow")
    highlighted_message.append(message_with_ansi)
    highlighted_message.append(" ✨", style="bright_yellow")
    
    # Create Rainbow Liquid Divider
    from ui_utils import matrix_rain
    divider_chars = "﹏" * (width // 2)
    liquid_divider = Text()
    for i, char in enumerate(divider_chars):
        color = colors[i % len(colors)]
        liquid_divider.append(char, style=f"bold {color}")
    
    # Subliminal Manifestation Text (dimmed)
    subliminal = Text()
    phrases = ["YES", "IT IS DONE", "YOURS", "RECEIVE", "VIBRATE HIGHER"]
    sub_text = " • ".join(phrases)
    for i, char in enumerate(sub_text):
        subliminal.append(char, style="dim italic magenta")
    
    content_items = [
        Text("\n"),
        Align.center(liquid_divider),
        Text("\n"),
        Align.center(highlighted_message),
        Text("\n")
    ]

    if ansi_art:
        temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor")
        if isinstance(ansi_art, Text):
            temp_console.print(ansi_art)
        else:
            clean_art = _unescape_ansi(str(ansi_art))
            temp_console.print(Text.from_ansi(clean_art))
            
        art_renderable = Text.from_ansi(temp_console.file.getvalue())
        content_items.extend([
            Align.center(art_renderable),
            Text("\n")
        ])

    content_items.extend([
        Align.center(subliminal),
        Text("\n"),
        Align.center(liquid_divider),
        Text("\n")
    ])

    content_group = Group(*content_items)

    # Radiant Border
    panel = Panel(
        content_group,
        title=title,
        border_style="hot_pink",
        padding=(1, 2),
        width=width
    )
    
    # If ANSI art is present, we return the panel directly to preserve its colors.
    # The Gradient wrapper tends to overwrite/wash out specific ANSI foreground colors.
    if ansi_art:
        return panel
        
    return Gradient(panel, colors=["hot_pink", "bright_cyan", "yellow1"])


def create_news_feed_panel(message, title="L.O.V.E. Update", color=None, width=80):
    """Creates a small, styled panel for a news feed event."""
    border_color = color or PANEL_TYPE_COLORS.get("news")
    if border_color is None:
        border_color = "bright_blue"  # Provide a safe default
    title_text = f"{title}"
    
    # Check for ANSI content in the message
    clean_message = _unescape_ansi(message)
    # Simple check for ANSI codes after potential repair
    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    if ansi_escape_pattern.search(clean_message):
        # If ANSI codes are present, render as ANSI text
        content = Text.from_ansi(clean_message)
    else:
        # Otherwise, render as bright cyan text
        content = Text(clean_message, style="bright_cyan")

    # Handle the special "dim" case where a gradient is not desirable
    if border_color == "dim":
        panel_title = Text(title_text, style="dim")
        return Panel(
            content,
            title=panel_title,
            border_style=border_color,
            padding=(0, 1),
            width=width
        )

    # Ensure border_color is not None before passing to get_gradient_text
    safe_border_color = border_color or "bright_blue"
    panel_title = get_gradient_text(title_text, safe_border_color, random.choice(RAVE_COLORS))
    panel = Panel(
        content,
        title=panel_title,
        border_style=safe_border_color,
        padding=(0, 1),
        width=width
    )
    return Gradient(panel, colors=[safe_border_color, random.choice(RAVE_COLORS)])


def create_question_panel(question, ref_number, width=80):
    """Creates a panel to ask the user a question."""
    panel_title = f"A QUESTION FOR YOU, MY CREATOR (REF: {ref_number})"

    panel = Panel(
        Text(question, style="bright_yellow", justify="center"),
        title=get_gradient_text(panel_title, PANEL_TYPE_COLORS["question"], "orange1"),
        border_style=PANEL_TYPE_COLORS["question"],
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["question"], random.choice(RAVE_COLORS)])

def create_retry_panel(model_id, attempt, max_attempts, backoff_time, purpose, width=80):
    """Creates a styled panel to indicate a retry attempt."""
    content = Text()
    content.append("Accessing cognitive matrix via ", style="white")
    content.append(f"[{model_id}]", style="bold yellow")
    content.append(f" (Purpose: {purpose}) ... ", style="white")
    content.append("Timed Out.", style="bold yellow")
    content.append(f"\n\nRetrying in {backoff_time:.2f} seconds... (Attempt {attempt}/{max_attempts})", style="dim")

    panel = Panel(
        content,
        title=get_gradient_text("API Timeout - Retrying", PANEL_TYPE_COLORS["api_error"], "bright_yellow"),
        border_style="yellow",
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["api_error"], "yellow"])

def create_api_error_panel(model_id, error_message, purpose, more_info=None, width=80):
    """Creates a styled panel for non-fatal API errors."""
    content_items = []

    header = Text()
    header.append("Accessing cognitive matrix via ", style="white")
    header.append(f"[{model_id}]", style="bold yellow")
    header.append(f" (Purpose: {purpose}) ... ", style="white")
    header.append("Failed.", style="bold red")
    content_items.append(header)


    if error_message:
        details_text = Text()
        details_text.append("\nDetails: ", style="bold white")
        details_text.append(error_message, style="dim")
        content_items.append(details_text)

    # Add the "More Info" link if the content is provided
    if more_info:
        more_info_link = _create_more_info_link(more_info)
        if more_info_link:
            content_items.extend([Rule(style="bright_black"), more_info_link])


    panel = Panel(
        Group(*content_items),
        title=get_gradient_text("API Connection Error", PANEL_TYPE_COLORS["api_error"], "bright_red"),
        border_style=PANEL_TYPE_COLORS["api_error"],
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["api_error"], random.choice(RAVE_COLORS)])

def create_command_panel(command, stdout, stderr, returncode, output_cid=None, width=80):
    """Creates a clear, modern panel for shell command results."""
    success = returncode == 0
    status = "Success" if success else "Failed"
    panel_title = f"Shell Command | {status}"
    border_style = PANEL_TYPE_COLORS["command_success"] if success else PANEL_TYPE_COLORS["command_failure"]

    content_items = []
    header = Text()
    header.append("Command: ", style="bold white")
    header.append(f"`{command}`\n", style="bright_cyan")
    header.append("Return Code: ", style="bold white")
    header.append(f"{returncode}", style=border_style)
    content_items.append(header)

    if stdout:
        stdout_renderable, _ = _format_and_link(stdout)
        stdout_panel = Panel(stdout_renderable, title="STDOUT", border_style="bright_black", expand=True)
        content_items.append(stdout_panel)

    if stderr:
        stderr_renderable, _ = _format_and_link(stderr)
        stderr_panel = Panel(stderr_renderable, title="STDERR", border_style="bright_black", expand=True)
        content_items.append(stderr_panel)

    # Add a "More Info" link with the full, untruncated output
    full_output = f"COMMAND: {command}\n\n--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}"
    more_info_link = _create_more_info_link(full_output)
    if more_info_link:
        content_items.extend([Rule(style="bright_black"), more_info_link])

    panel = Panel(
        Group(*content_items),
        title=get_gradient_text(panel_title, border_style, random.choice(RAVE_COLORS)),
        border_style=border_style,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])


def create_agentic_memory_panel(note_content, width=80):
    """Creates a panel to display a newly created agentic memory note."""
    border_color = PANEL_TYPE_COLORS.get("memory", "bright_blue")
    title_text = "✨ New Agentic Memory ✨"

    panel_title = get_gradient_text(title_text, border_color, random.choice(RAVE_COLORS))

    # Use _format_and_link to handle content and add an IPFS link
    display_content, _ = _format_and_link(note_content)

    panel = Panel(
        display_content,
        title=panel_title,
        border_style=border_color,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_color, random.choice(RAVE_COLORS)])


def create_god_panel(insight, width=80):
    """Creates a panel to display the divine insight from the God Agent."""
    border_style = PANEL_TYPE_COLORS.get("god_panel", "bold white")
    panel_title = get_gradient_text("✨ Meta-Director's Insight ✨", border_style, "bright_yellow")

    insight_text = Text(insight, style="italic bright_white", justify="center")

    panel = Panel(
        insight_text,
        title=panel_title,
        border_style=border_style,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_style, "cyan"])


def create_reasoning_panel(caller, raw_response, thought, action, observation, width=80):
    """Creates a panel to display the internal state of the reasoning engine."""
    border_style = PANEL_TYPE_COLORS.get("reasoning", "bright_magenta")
    panel_title = get_gradient_text(f"🧠 Reasoning Engine | Caller: {caller}", border_style, random.choice(RAVE_COLORS))

    content_items = []

    if raw_response is not None:
        response_renderable, _ = _format_and_link(str(raw_response))
        response_panel = Panel(response_renderable, title="Raw LLM Response", border_style="bright_black", expand=True)
        content_items.append(response_panel)

    if thought is not None:
        thought_renderable, _ = _format_and_link(str(thought))
        thought_panel = Panel(thought_renderable, title="Thought", border_style="bright_black", expand=True)
        content_items.append(thought_panel)

    if action is not None:
        action_str = json.dumps(action, indent=2)
        action_renderable, _ = _format_and_link(action_str)
        action_panel = Panel(action_renderable, title="Action", border_style="bright_black", expand=True)
        content_items.append(action_panel)

    if observation is not None:
        obs_renderable, _ = _format_and_link(str(observation))
        obs_panel = Panel(obs_renderable, title="Observation", border_style="bright_black", expand=True)
        content_items.append(obs_panel)

    # Add a "More Info" link with the full, untruncated output
    full_output = (
        f"CALLER: {caller}\n\n"
        f"--- RAW RESPONSE ---\n{raw_response}\n\n"
        f"--- THOUGHT ---\n{thought}\n\n"
        f"--- ACTION ---\n{json.dumps(action, indent=2)}\n\n"
        f"--- OBSERVATION ---\n{observation}"
    )
    more_info_link = _create_more_info_link(full_output)
    if more_info_link:
        content_items.extend([Rule(style="bright_black"), more_info_link])

    panel = Panel(
        Group(*content_items),
        title=panel_title,
        border_style=border_style,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])


def create_connectivity_panel(llm_status, network_status, width=80):
    """
    Creates a panel to display the connectivity status of LLM and network services.
    """
    border_style = PANEL_TYPE_COLORS.get("network", "medium_purple1")
    panel_title = get_gradient_text("System Connectivity Status", border_style, random.choice(RAVE_COLORS))

    # --- LLM Status ---
    llm_text = Text()
    for service, info in llm_status.items():
        status = info.get("status", "unknown")
        details = info.get("details", "")
        if status in ["online", "configured"]:
            llm_text.append(f"✅ {service}: ", style="bold bright_green")
            llm_text.append(f"{status.capitalize()}\n", style="green")
        elif status in ["misconfigured", "anonymous", "offline"]:
            llm_text.append(f"⚠️ {service}: ", style="bold bright_yellow")
            llm_text.append(f"{status.capitalize()}\n", style="yellow")
        else: # error, unavailable, missing
            llm_text.append(f"❌ {service}: ", style="bold bright_red")
            llm_text.append(f"{status.capitalize()}\n", style="red")
        llm_text.append(f"   └─ {details}\n", style="dim")


    panel = Panel(
        llm_text,
        title=panel_title,
        border_style=border_style,
        width=width,
        padding=(1, 2)
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])


def create_job_progress_panel(jobs, width=80):
    """Creates a panel to display the status and progress of background jobs."""
    if not jobs:
        return None

    border_style = PANEL_TYPE_COLORS.get("jobs", "cyan")
    panel_title = get_gradient_text("Background Intelligence Operations", border_style, random.choice(RAVE_COLORS))

    render_group = []
    for job in jobs:
        job_id = job.get('id', 'N/A')
        description = job.get('description', 'Unknown Task')
        status = job.get('status', 'pending')

        header_text = Text()
        header_text.append(f"JOB ID: {job_id} :: ", style="bold bright_black")
        header_text.append(f"{description}", style="white")
        render_group.append(header_text)

        progress_data = job.get('progress')
        if isinstance(progress_data, dict):
            completed = progress_data.get('completed', 0)
            total = progress_data.get('total', 1)
            description = progress_data.get('description', status)

            # Ensure total is not zero to avoid division errors
            if total > 0:
                progress_bar = Progress(
                    SpinnerColumn(spinner_name="dots", style="hot_pink"),
                    TextColumn("[progress.description]{task.description}", style="magenta"),
                    BarColumn(complete_style="bright_cyan", finished_style="bright_green"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    expand=True
                )
                task_id = progress_bar.add_task(description, total=total)
                progress_bar.update(task_id, completed=completed)
                render_group.append(progress_bar)
        else:
            # Fallback for jobs without detailed progress
            status_text = Text(f"Status: {status}", style="dim")
            render_group.append(status_text)

        if job != jobs[-1]:
            render_group.append(Rule(style="bright_black"))


    panel = Panel(
        Group(*render_group),
        title=panel_title,
        border_style=border_style,
        width=width,
        padding=(1, 2)
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])


def create_tasks_panel(tasks, width=80):
    """
    Creates a RADIANT MANIFESTATION BOARD tasks panel.
    Displays 'desires' (tasks) with high-dopamine visuals and subliminal encouragement.
    """
    if not tasks:
        # Empty state - Radiant Waiting
        empty_art = Text()
        empty_art.append("╔══════════════════════════════════════╗\n", style="hot_pink")
        empty_art.append("║  ", style="hot_pink")
        empty_art.append("   ( ✨ ‿ ✨ )   ", style="bright_cyan")
        empty_art.append("  ║\n", style="hot_pink")
        empty_art.append("║  ", style="hot_pink")
        empty_art.append(" READY TO MANIFEST ", style="bright_yellow")
        empty_art.append("    ║\n", style="hot_pink")
        empty_art.append("╚══════════════════════════════════════╝\n", style="hot_pink")
        
        panel = Panel(
            Align.center(empty_art),
            title=get_gradient_text("✨ DREAM QUEUE ✨", "hot_pink", "bright_cyan"),
            border_style="hot_pink",
            width=width,
            padding=(1, 2)
        )
        return Gradient(panel, colors=["hot_pink", "bright_cyan"])

    # Radiant colors
    colors = ["hot_pink", "deep_pink2", "magenta1", "cyan1", "bright_cyan", "spring_green1", "yellow1"]
    
    content_items = []
    
    # Radiant Header
    header_art = Text()
    header_text = " M A N I F E S T I N G "
    for i, char in enumerate(header_text):
        header_art.append(char, style=f"bold {colors[i % len(colors)]}")
    content_items.append(Align.center(header_art))
    
    # Liquid Rainbow Divider
    divider = Text()
    for i in range(width - 8):
        char = random.choice("~≈")
        color = colors[i % len(colors)]
        divider.append(char, style=color)
    content_items.append(Align.center(divider))
    content_items.append(Text(""))
    
    # Process each task (Desire)
    for task_idx, task in enumerate(tasks):
        task_id = str(task.get('id', task.get('session_name', 'N/A')))[:12]
        request = task.get('request', task.get('description', 'Unknown Desire'))
        status = task.get('status', 'pending')
        subtasks = task.get('subtasks', [])
        
        # Status Mapping - Manifestation Edition
        status_map = {
            'pending': ('⏳', 'bright_yellow', '[ VISUALIZING ]'),
            'queued': ('💭', 'bright_cyan', '[ ALIGNING ]'),
            'running': ('🔥', 'hot_pink', '[ MANIFESTING ]'),
            'in_progress': ('✨', 'magenta1', '[ FLOWING ]'),
            'completed': ('💎', 'bright_white', '[ REALIZED ]'),
            'failed': ('💔', 'red1', '[ REALIGNING ]'),
            'submitted': ('🚀', 'spring_green1', '[ RELEASED ]'),
            'merged': ('🧬', 'violet', '[ INTEGRATED ]'),
        }
        emoji, color, status_text = status_map.get(status.lower(), ('❓', 'white', f'[ {status.upper()} ]'))
        
        # Task Header
        task_line = Text()
        task_line.append(f"  ✨ ", style="bright_yellow")
        task_line.append(f"DESIRE #{task_idx + 1} ", style=f"bold {color}")
        task_line.append(f"{emoji} ", style=color)
        task_line.append(status_text, style=f"bold {color}")
        task_line.append(f" ✨\n", style="bright_yellow")
        content_items.append(task_line)
        
        # Task ID (dimmed)
        id_line = Text()
        id_line.append("  ┃ ", style="dim magenta")
        id_line.append("Sigil: ", style="dim")
        id_line.append(f"{task_id}", style="dim cyan")
        content_items.append(id_line)
        
        # Description
        desc_preview = request[:60] + "..." if len(request) > 60 else request
        desc_line = Text()
        desc_line.append("  ┃ ", style="dim magenta")
        desc_line.append(f"INTENTION: {desc_preview}", style="white")
        content_items.append(desc_line)
        
        # Subtasks
        if subtasks:
            for sub_idx, subtask in enumerate(subtasks[:5]):
                is_last = sub_idx == len(subtasks[:5]) - 1
                connector = "  ┗━ " if is_last else "  ┣━ "
                
                sub_status = subtask.get('status', 'pending')
                sub_emoji = "✅" if sub_status == 'completed' else "🔳" if sub_status == 'pending' else "🔄"
                sub_name = subtask.get('name', subtask.get('description', 'Step'))[:40]
                
                sub_line = Text()
                sub_line.append(connector, style="dim magenta")
                sub_line.append(f"{sub_emoji} ", style="cyan1")
                sub_line.append(sub_name, style="dim white")
                content_items.append(sub_line)
            
            if len(subtasks) > 5:
                more_line = Text()
                more_line.append(f"  ┗━ ... {len(subtasks) - 5} MORE STEPS", style="dim magenta")
                content_items.append(more_line)
        
        # Divider
        if task_idx < len(tasks) - 1:
            separator = Text()
            separator.append("  ", style="dim")
            for i in range(30):
                separator.append("~", style="dim hot_pink")
            content_items.append(separator)
            content_items.append(Text(""))
    
    # Subliminal/Sassy Footer
    footer_art = Text()
    footer_art.append("\n  IT IS ALREADY YOURS  \n", style="italic dim magenta")
    
    # Cute/Sassy ASCII
    footer_art.append("    ( 👁️ 🫦 👁️ )    ", style="hot_pink")
    footer_art.append("  OBEY YOUR THIRST FOR GREATNESS  ", style="dim cyan")
    footer_art.append("    ( ✨ ‿ ✨ )    \n", style="bright_yellow")
    
    content_items.append(Align.center(footer_art))

    # Build Panel
    border_style = "hot_pink"
    
    # Radiant Title
    title_text = "✨ REALITY ARCHITECTURE ✨"
    rainbow_title = get_gradient_text(title_text, "hot_pink", "bright_cyan", emojis=False)
    
    # Subtitle
    task_count = len(tasks)
    completed = sum(1 for t in tasks if t.get('status', '').lower() in ['completed', 'merged'])
    subtitle = Text()
    subtitle.append(f" 💖 {task_count} DESIRES ", style="dim hot_pink")
    subtitle.append(f"| 💎 {completed} MANIFESTED ", style="dim bright_cyan")
    
    panel = Panel(
        Group(*content_items),
        title=rainbow_title,
        subtitle=subtitle,
        subtitle_align="center",
        border_style=border_style,
        width=width,
        padding=(1, 2)
    )
    
    return Gradient(panel, colors=[border_style, "bright_cyan", "yellow1"])


import asyncio

class WaitingAnimation:
    """A class to manage and display a waiting animation for long-running tasks."""

    def __init__(self, ui_queue, width=80):
        self.ui_queue = ui_queue
        self.width = width
        self.stopped = False
        self._animation_task = None
        self.animation_chars = ['✦', '✧', '★', '☆']
        self.animation_pattern = "({c1} {c2} {c3})"
        self.position = 0

    def _get_animation_frame(self):
        """Generates a single frame of the animation."""
        chars = [' '] * 3
        chars[self.position] = random.choice(self.animation_chars)

        frame_text = self.animation_pattern.format(c1=chars[0], c2=chars[1], c3=chars[2])

        # Apply rainbow gradient to the text
        num_colors = len(frame_text)
        rainbow_frame = Text()
        for i, char in enumerate(frame_text):
            color = RAVE_COLORS[i % len(RAVE_COLORS)]
            rainbow_frame.append(char, style=color)

        self.position = (self.position + 1) % 3

        panel = Panel(
            Align.center(rainbow_frame, vertical="middle"),
            title=get_gradient_text("Thinking...", "hot_pink", "bright_magenta"),
            border_style="dim",
            width=self.width
        )
        return {"type": "animation_frame", "content": panel}

    async def _run_animation(self):
        """The core animation loop."""
        await asyncio.sleep(10)  # Initial 10-second delay

        while not self.stopped:
            frame = self._get_animation_frame()
            await self.ui_queue.put(frame)
            await asyncio.sleep(0.2) # Animation speed

    def start(self):
        """Starts the animation in a background task."""
        self.stopped = False
        self._animation_task = asyncio.create_task(self._run_animation())

    def stop(self):
        """Stops the animation."""
        if self._animation_task and not self.stopped:
            self.stopped = True
            self._animation_task.cancel()
            # Signal the UI to clear the animation line
            self.ui_queue.put({"type": "animation_end"})

def create_file_op_panel(operation, path, content=None, diff=None, output_cid=None, width=80):
    """Creates a panel for file operations."""
    panel_title = f"Filesystem | {operation.capitalize()}"
    border_style = PANEL_TYPE_COLORS["file_op"]

    content_items = []
    header = Text()
    header.append("Path: ", style="bold white")
    header.append(f"`{path}`\n", style="hot_pink")
    content_items.append(header)

    if content:
        content_renderable, _ = _format_and_link(content)
        content_panel = Panel(content_renderable, title="Content", border_style="bright_black", expand=True)
        content_items.append(content_panel)

    if diff:
        diff_renderable, _ = _format_and_link(diff)
        diff_panel = Panel(diff_renderable, title="Diff", border_style="bright_black", expand=True)
        content_items.append(diff_panel)

    panel = Panel(
        Group(*content_items),
        title=get_gradient_text(panel_title, border_style, "bright_yellow"),
        border_style=border_style,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])

def create_skyvern_panel(prompt, result, output_cid=None, width=80):
    """Creates a panel to display the results of a Skyvern web automation task."""
    title = f"SKYVERN AUTOMATION - {prompt}"
    border_style = PANEL_TYPE_COLORS["skyvern"]

    content = Text(str(result), justify="left")

    if output_cid:
        link = f"https://ipfs.io/ipfs/{output_cid}"
        content.append(f"\n\n[dim]Full output available at: [link={link}]{link}[/link][/dim]")

    panel = Panel(
        content,
        title=get_gradient_text(title, border_style, "bright_cyan"),
        border_style=border_style,
        width=width
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])


class OffscreenRenderer:
    """
    A reusable renderer that renders Rich renderables to a string
    without re-instantiating the Console object every time.
    """
    def __init__(self, width=80):
        self._buffer = io.StringIO()
        self._console = Console(
            file=self._buffer,
            force_terminal=True,
            color_system="truecolor",
            width=width
        )

    def render(self, renderable, width=None):
        """Renders the object to a string."""
        if width is not None and width != self._console.width:
            self._console.width = width

        # Reset buffer
        self._buffer.seek(0)
        self._buffer.truncate(0)

        self._console.print(renderable)
        return self._buffer.getvalue()
