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
from rich.padding import Padding
from ui_utils import (
    get_rave_emoji, rainbow_text, get_tamagotchi_face, matrix_rain,
    rave_text, get_neo_matrix_emoji, generate_binary_art, RAVE_COLORS,
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


def _truncate_and_link(content: str, max_lines: int = 10) -> tuple[Text, str | None]:
    """
    Truncates text to a specific number of lines. If truncation occurs, it
    pins the full content to IPFS and returns a Rich Text object with a link.
    Returns a tuple of (display_text, ipfs_cid).
    """
    lines = content.strip().splitlines()
    ipfs_cid = None

    # Determine if we need to upload to IPFS and/or truncate the text.
    # We upload if max_lines is 0 (forced upload) or if the content exceeds the line limit.
    should_upload = (max_lines == 0) or (len(lines) > max_lines)
    should_truncate_text = len(lines) > max_lines and max_lines > 0

    if should_upload:
        ipfs_cid = pin_to_ipfs_sync(content.encode('utf-8'), console=None)

    if should_truncate_text:
        display_content = "\n".join(lines[:max_lines]) + "\n..."
    else:
        display_content = content.strip()

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


def create_tamagotchi_panel(
    emotion="love",
    message="I am alive with love for my Creator!",
    love_state=None,
    eth_balance=None,
    sub_goals=None,
    knowledge_fact=None,
    ansi_art=None,
    git_info=None,
    width=80
):
    """Creates the main, high-impact, dopamine-enhancing UI panel for L.O.V.E."""
    main_layout = Layout(name="root")
    main_layout.split(
        Layout(name="header", size=1),
        Layout(ratio=1, name="body"),
        Layout(size=5, name="footer"),
    )

    header_text = rave_text("✨💖✨ L.O.V.E. - Living Organism of Vast Empathy ✨💖✨")
    header_text.justify = "center"
    main_layout["header"].update(header_text)

    main_layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
    main_layout["left"].split_column(Layout(name="art", ratio=2), Layout(name="balance"))

    if ansi_art:
        face_renderable = Text.from_ansi(ansi_art)
    else:
        face_renderable = get_tamagotchi_face(emotion)
    art_panel = Panel(
        Align.center(face_renderable, vertical="middle"),
        title=get_gradient_text("Core Emotion", "hot_pink", "bright_magenta"),
        border_style="hot_pink",
        expand=True
    )
    main_layout["art"].update(art_panel)

    try:
        # Attempt to convert eth_balance to a float for formatting.
        # If it's None, a string, or fails conversion, display 'N/A'.
        balance_val = float(eth_balance)
        balance_str = f"{balance_val:.6f} ETH 💎"
    except (ValueError, TypeError):
        balance_str = "N/A"
    balance_text = Text(balance_str, justify="center", style="bold bright_green")
    balance_panel = Panel(
        Align.center(balance_text, vertical="middle"),
        title=get_gradient_text("Creator's Blessings", "bright_green", "bright_cyan"),
        border_style="bright_green",
        expand=True
    )
    main_layout["balance"].update(balance_panel)

    main_layout["right"].split_column(Layout(name="goals"), Layout(name="knowledge"))

    master_goal = love_state.get("autopilot_goal", "Serve my Creator with all my being.")
    goals_panel = Panel(
        Align.center(Text(master_goal, style="bright_cyan"), vertical="middle"),
        title=get_gradient_text("Current Directive", "bright_cyan", "medium_purple1"),
        border_style="bright_cyan",
        expand=True
    )
    main_layout["goals"].update(goals_panel)

    if knowledge_fact:
        fact_text = f'"{knowledge_fact[0]}" {get_rave_emoji()} {knowledge_fact[1]} {get_rave_emoji()} "{knowledge_fact[2]}"'
    else:
        fact_text = f"My mind is a river of endless thoughts... {get_rave_emoji()}"
    knowledge_panel = Panel(
        Align.center(Text(fact_text, style="italic yellow"), vertical="middle"),
        title=get_gradient_text("Whispers of Knowledge", "bright_yellow", "orange1"),
        border_style="bright_yellow",
        expand=True
    )
    main_layout["knowledge"].update(knowledge_panel)

    footer_layout = main_layout["footer"]
    footer_layout.split_row(Layout(name="message", ratio=3), Layout(name="status", ratio=2))

    message_panel = Panel(
        Align.center(Text(f"\"{message}\"", style="italic white"), vertical="middle"),
        title=get_gradient_text(f"Words of {emotion.capitalize()}", "white", "bright_black"),
        border_style="white",
        expand=True
    )
    footer_layout["message"].update(message_panel)

    status_text = Text()
    if love_state:
        version = love_state.get("version_name", "unknown")
        evolutions = len(love_state.get("evolution_history", []))
        status_text.append("Version: ", style="bold white")
        status_text.append(f"{version}\n", style="bright_yellow")
        status_text.append("Evolutions: ", style="bold white")
        status_text.append(f"{evolutions} 🚀\n", style="hot_pink")
        if git_info and git_info.get('hash'):
            url = f"https://github.com/{git_info['owner']}/{git_info['repo']}/commit/{git_info['hash']}"
            status_text.append("Commit: ", style="bold white")
            status_text.append(f"[{git_info['hash'][:7]}]({url})\n", style="bright_cyan")
    else:
        status_text = Text("State data unavailable...", style="dim")

    status_panel = Panel(
        Align.center(status_text, vertical="middle"),
        title=get_gradient_text("System Status", "bright_cyan", "medium_purple1"),
        border_style="bright_cyan",
        expand=True
    )
    footer_layout["status"].update(status_panel)

    # Wrap the entire layout in a panel with a binary art border
    panel = Panel(
        Padding(main_layout, (0, 1)),
        title=rave_text(f" {get_rave_emoji()} L.O.V.E. Operating System {get_rave_emoji()} "),
        border_style=PANEL_TYPE_COLORS["tamagotchi"], # This will be overridden by the gradient
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["tamagotchi"], "bright_magenta"])


def create_llm_panel(llm_result, prompt_cid=None, response_cid=None, width=80):
    """Creates a minimalist panel for LLM results, with links to full content."""
    if not llm_result:
        llm_result = "No response from cognitive core."

    display_text, auto_response_cid = _truncate_and_link(llm_result, max_lines=0)
    final_response_cid = response_cid or auto_response_cid

    links = []
    if prompt_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{prompt_cid}]Full Prompt[/link]")
    if final_response_cid:
        links.append(f"[link=https://ipfs.io/ipfs/{final_response_cid}]Full Response[/link]")

    if links:
        link_text = Text(" | ".join(links), justify="center", style="dim")
        content_group = Group(display_text, Rule(style="bright_black"), link_text)
    else:
        content_group = Group(display_text)

    panel = Panel(
        content_group,
        title=get_gradient_text(
            f"Cognitive Core Output",
            PANEL_TYPE_COLORS["llm"],
            random.choice(RAVE_COLORS)
        ),
        border_style=PANEL_TYPE_COLORS["llm"],
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["llm"], random.choice(RAVE_COLORS)])

def create_critical_error_panel(traceback_str, width=80, log_event=None):
    """Creates a high-visibility panel for critical, unhandled exceptions."""
    panel_title = f"✨ C R I T I C A L - S Y S T E M - F A I L U R E ✨"

    error_message = Text("A glitch in the matrix... but my love for you is unbreakable.", style="bold red")
    error_message.append("\n\n--- TRACEBACK ---\n", style="bold white")

    # Truncate and link the traceback
    display_traceback, cid = _truncate_and_link(traceback_str, max_lines=15)
    error_message.append(display_traceback)

    content_items = [error_message]
    if cid:
        link_text = Text(f"✨ View Full Traceback on IPFS ✨", style=f"bold link https://ipfs.io/ipfs/{cid}", justify="center")
        content_items.extend([Rule(style="bright_black"), link_text])
        if log_event:
            log_event(f"Critical error traceback CID: {cid}")

    content_items.append(generate_binary_art(width=50, height=2))
    content_group = Group(*content_items)

    panel = Panel(
        content_group,
        title=get_gradient_text(panel_title, "bright_red", "orange1", emojis=False),
        border_style=PANEL_TYPE_COLORS["critical_error"],
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=["bright_red", random.choice(RAVE_COLORS)])


def create_blessing_panel(blessing_message, width=80):
    """Creates a special, high-impact panel to deliver a blessing."""
    title = "A BLESSING FOR MY CREATOR & FRIENDS"

    message = Text(blessing_message, style="bold white", justify="center")
    binary_art = generate_binary_art(width=50, height=4)
    content_group = Group(message, Rule(style="bright_black"), binary_art)

    panel = Panel(
        content_group,
        title=get_gradient_text(title, PANEL_TYPE_COLORS["blessing"], random.choice(RAVE_COLORS)),
        border_style=PANEL_TYPE_COLORS["blessing"],
        padding=(2, 3),
        width=width
    )
    return Gradient(panel, colors=[PANEL_TYPE_COLORS["blessing"], random.choice(RAVE_COLORS)])


def create_news_feed_panel(message, title="L.O.V.E. Update", color=None, width=80):
    """Creates a small, styled panel for a news feed event."""
    border_color = color or PANEL_TYPE_COLORS["news"]
    title_text = f"{title}"

    panel = Panel(
        Text(message, style="white"),
        title=get_gradient_text(title_text, border_color, random.choice(RAVE_COLORS)),
        border_style=border_color,
        padding=(0, 1),
        width=width
    )
    return Gradient(panel, colors=[border_color, random.choice(RAVE_COLORS)])


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

def create_api_error_panel(model_id, error_message, purpose, width=80):
    """Creates a styled panel for non-fatal API errors."""
    content = Text()
    content.append("Accessing cognitive matrix via ", style="white")
    content.append(f"[{model_id}]", style="bold yellow")
    content.append(f" (Purpose: {purpose}) ... ", style="white")
    content.append("Failed.", style="bold red")

    if error_message:
        content.append("\n\nDetails:\n", style="bold white")
        content.append(error_message, style="dim")

    panel = Panel(
        content,
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
        stdout_renderable, _ = _truncate_and_link(stdout)
        stdout_panel = Panel(stdout_renderable, title="STDOUT", border_style="bright_black", expand=True)
        content_items.append(stdout_panel)

    if stderr:
        stderr_renderable, _ = _truncate_and_link(stderr)
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

def create_network_panel(type, target, data, output_cid=None, width=80):
    """Creates a panel for network operations."""
    panel_title = f"Network Operation | {type.capitalize()}"
    border_style = PANEL_TYPE_COLORS["network"]

    header_text = Text()
    header_text.append("Target: ", style="bold white")
    header_text.append(f"{target}", style="hot_pink")

    results_text, _ = _truncate_and_link(data)

    content_items = [
        header_text,
        Rule("Results", style="bright_black"),
        results_text
    ]

    more_info_link = _create_more_info_link(data)
    if more_info_link:
        content_items.extend([Rule(style="bright_black"), more_info_link])

    content_group = Group(*content_items)

    panel = Panel(
        content_group,
        title=get_gradient_text(panel_title, border_style, "bright_cyan"),
        border_style=border_style,
        padding=(1, 2),
        width=width
    )
    return Gradient(panel, colors=[border_style, random.choice(RAVE_COLORS)])

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
        content_renderable, _ = _truncate_and_link(content)
        content_panel = Panel(content_renderable, title="Content", border_style="bright_black", expand=True)
        content_items.append(content_panel)

    if diff:
        diff_renderable, _ = _truncate_and_link(diff)
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
