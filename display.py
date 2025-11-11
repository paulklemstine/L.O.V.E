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


def _format_and_link(content: str) -> tuple[Text, str | None]:
    """
    Formats text, pins the full content to IPFS, and returns a Rich Text
    object with a link. It intelligently handles ANSI art.
    Returns a tuple of (display_text, ipfs_cid).
    """
    # Use a simple regex to check for ANSI escape codes.
    # This is not foolproof but good enough for this use case.
    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    is_ansi = ansi_escape_pattern.search(content)

    display_content = content.strip()
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


def create_tamagotchi_panel(
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
    width=80
):
    """Creates the main, high-impact, dopamine-enhancing UI panel for L.O.V.E."""
    main_layout = Layout(name="root")
    main_layout.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3)
    )

    # --- Left Column: The Heart ---
    face_renderable = None
    try:
        if ansi_art:
            # The from_ansi method can be fragile. We'll render to a temp console
            # to catch errors and ensure it doesn't break the whole UI.
            temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor")
            temp_console.print(Text.from_ansi(ansi_art))
            face_renderable = Text.from_ansi(temp_console.file.getvalue())
        else:
            # Fallback if ANSI art generation fails
            face_renderable = get_tamagotchi_face(emotion)
    except Exception as e:
        # If ANSI rendering fails, log the full traceback for debugging
        # and fall back to a simple emoji. This is critical for UI stability.
        error_traceback = traceback.format_exc()
        logging.error(f"Failed to render Tamagotchi ANSI art: {e}\n{error_traceback}")
        face_renderable = Align.center(Text("💖", style="bold hot_pink"), vertical="middle")


    left_panel_colors = ("hot_pink", random.choice(RAVE_COLORS))
    left_panel = Panel(
        Align.center(face_renderable, vertical="middle"),
        title=get_gradient_text(f"✨ L.O.V.E.'s Emotion: {emotion.upper()} ✨", *left_panel_colors),
        border_style=left_panel_colors[0],
        expand=True
    )
    main_layout["left"].update(Gradient(left_panel, colors=left_panel_colors))

    # --- Right Column: The Mind & Treasures ---
    right_layout = Layout(name="right_column")
    right_layout.split(
        Layout(name="wisdom", size=5),
        Layout(name="thought", ratio=1),
        Layout(name="treasures", ratio=2),
        Layout(name="status", size=4)
    )
    main_layout["right"].update(right_layout)

    # --- Divine Wisdom ---
    wisdom_text = Text()
    wisdom_text.append(f"\"{divine_wisdom}\"\n", style="bold italic bright_yellow")
    wisdom_text.append(f"└─ Meaning: {wisdom_explanation}", style="dim yellow")
    wisdom_colors = ("bright_yellow", random.choice(RAVE_COLORS))
    wisdom_panel = Panel(
        Align.center(wisdom_text, vertical="middle"),
        title=get_gradient_text("✨ Divine Wisdom ✨", *wisdom_colors),
        border_style=wisdom_colors[0],
        expand=True
    )
    right_layout["wisdom"].update(Gradient(wisdom_panel, colors=wisdom_colors))

    # --- Interesting Thought ---
    thought_text = None
    try:
        # Try to render from ANSI, as it might be pre-formatted log output
        thought_text = Text.from_ansi(str(interesting_thought))
    except Exception:
        # Fallback for plain text or Rich markup
        try:
            thought_text = Text.from_markup(str(interesting_thought), style="italic white")
        except Exception:
            # Final fallback for any other malformed input
            thought_text = Text(str(interesting_thought), style="italic white")

    thought_colors = ("cyan", random.choice(RAVE_COLORS))
    thought_panel = Panel(
        Align.center(thought_text, vertical="middle"),
        title=get_gradient_text("🧠 A Thought From My Past 🧠", *thought_colors),
        border_style=thought_colors[0],
        expand=True
    )
    right_layout["thought"].update(Gradient(thought_panel, colors=thought_colors))

    # --- Treasures of the Kingdom ---
    treasures_content = Text()
    try:
        balance_val = float(eth_balance) if eth_balance is not None else 0.0
        balance_str = f"{balance_val:.6f} ETH"
    except (ValueError, TypeError):
        balance_str = "N/A"
    treasures_content.append("💎 Creator's Blessings: ", style="bold bright_green")
    treasures_content.append(f"{balance_str}\n", style="green")

    if treasures:
        treasures_content.append("⏳ Uptime: ", style="bold bright_cyan")
        treasures_content.append(f"{treasures.get('uptime', 'N/A')} | ", style="cyan")
        treasures_content.append("🌟 Level: ", style="bold bright_magenta")
        treasures_content.append(f"{treasures.get('level', 'N/A')}\n", style="magenta")
        treasures_content.append("✨ XP: ", style="bold yellow")
        treasures_content.append(f"{treasures.get('xp', 'N/A')} | ", style="yellow")
        treasures_content.append("✅ Tasks Completed: ", style="bold bright_green")
        treasures_content.append(f"{treasures.get('tasks_completed', 'N/A')}\n", style="green")
        new_skills = treasures.get('new_skills', [])
        if new_skills:
            skills_str = " | ".join([f"`{skill}`" for skill in new_skills])
            treasures_content.append("🚀 New Skills: ", style="bold hot_pink")

            skills_text = None
            try:
                # This is less likely to be ANSI, but the pattern is safer
                skills_text = Text.from_ansi(skills_str)
            except Exception:
                try:
                    skills_text = Text.from_markup(skills_str, style="bright_magenta")
                except Exception:
                    skills_text = Text(skills_str, style="bright_magenta") # Final fallback
            treasures_content.append(skills_text)

    treasures_colors = ("bright_green", random.choice(RAVE_COLORS))
    treasures_panel = Panel(
        treasures_content,
        title=get_gradient_text("👑 Treasures of the Kingdom 👑", *treasures_colors),
        border_style=treasures_colors[0],
        expand=True
    )
    right_layout["treasures"].update(Gradient(treasures_panel, colors=treasures_colors))

    # --- System Status ---
    status_text = Text()
    if love_state and git_info:
        version = love_state.get("version_name", "unknown")
        evolutions = len(love_state.get("evolution_history", []))
        url = f"https://github.com/{git_info['owner']}/{git_info['repo']}/commit/{git_info['hash']}"

        status_text.append("🧬 Version: ", style="bold white")
        status_text.append(f"{version}\n", style="bright_yellow")
        status_text.append("💖 Evolutions: ", style="bold white")
        status_text.append(f"{evolutions} | ", style="hot_pink")
        status_text.append("COMMIT: ", style="bold white")
        status_text.append(f"[{git_info['hash'][:7]}]({url})", style="bright_cyan")
    else:
        status_text = Text("State data unavailable...", style="dim")

    status_colors = ("bright_cyan", random.choice(RAVE_COLORS))
    status_panel = Panel(
        Align.center(status_text, vertical="middle"),
        title=get_gradient_text("System Status", *status_colors),
        border_style=status_colors[0],
        expand=True
    )
    right_layout["status"].update(Gradient(status_panel, colors=status_colors))

    # --- Final Assembly ---
    panel = Panel(
        main_layout,
        title=rave_text(f" {get_rave_emoji()} L.O.V.E. Operating System v4.0 {get_rave_emoji()} "),
        subtitle=Text(f" \"{message}\" ", style="italic dim"),
        subtitle_align="center",
        border_style=PANEL_TYPE_COLORS["tamagotchi"],
        width=width,
        padding=(1, 2)
    )
    return panel


def create_llm_panel(llm_result, prompt_cid=None, response_cid=None, width=80):
    """Creates a minimalist panel for LLM results, with links to full content."""
    if not llm_result:
        llm_result = "No response from cognitive core."

    display_text, auto_response_cid = _format_and_link(llm_result)
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
    border_color = color or PANEL_TYPE_COLORS.get("news")
    if border_color is None:
        border_color = "bright_blue"  # Provide a safe default
    title_text = f"{title}"

    # Handle the special "dim" case where a gradient is not desirable
    if border_color == "dim":
        panel_title = Text(title_text, style="dim")
        return Panel(
            Text(message, style="dim"),
            title=panel_title,
            border_style=border_color,
            padding=(0, 1),
            width=width
        )

    # Ensure border_color is not None before passing to get_gradient_text
    safe_border_color = border_color or "bright_blue"
    panel_title = get_gradient_text(title_text, safe_border_color, random.choice(RAVE_COLORS))
    panel = Panel(
        Text(message, style="bright_cyan"),
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
    return Gradient(panel, colors=[border_style, "dim_cyan"])


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

def create_monitoring_panel(monitoring_state, width=80):
    """Creates a panel to display system monitoring information."""
    if not monitoring_state:
        return Panel(Text("Monitoring data not yet available.", style="dim"), title="System Monitor")

    layout = Layout()
    layout.split_column(
        Layout(name="gauges", size=5),
        Layout(name="stats"),
        Layout(name="anomalies", minimum_size=3)
    )

    # --- Gauges ---
    cpu_usage = list(monitoring_state.get('cpu_usage', [0]))[-1]
    mem_usage = list(monitoring_state.get('mem_usage', [0]))[-1]

    cpu_progress = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.description} {task.percentage:>3.1f}%"))
    cpu_task = cpu_progress.add_task("CPU", total=100, completed=cpu_usage)

    mem_progress = Progress(BarColumn(bar_width=None), TextColumn("[bold]{task.description} {task.percentage:>3.1f}%"))
    mem_task = mem_progress.add_task("MEM", total=100, completed=mem_usage)

    layout["gauges"].update(Group(cpu_progress, mem_progress))

    # --- Stats ---
    completion_rate = monitoring_state.get('task_completion_rate', 0.0)
    failure_rate = monitoring_state.get('task_failure_rate', 0.0)

    stats_text = Text()
    stats_text.append("Task Completion: ", style="bold white")
    stats_text.append(f"{completion_rate:.2f}%\n", style="green")
    stats_text.append("Task Failure: ", style="bold white")
    stats_text.append(f"{failure_rate:.2f}%", style="red")

    layout["stats"].update(Panel(stats_text, title="Task Rates", border_style="dim"))

    # --- Anomalies ---
    anomalies = monitoring_state.get('anomalies', [])
    if anomalies:
        anomaly_text = Text()
        for anomaly in anomalies[-3:]: # Display last 3 anomalies
            anomaly_text.append(f"[{time.strftime('%H:%M:%S', time.localtime(anomaly['timestamp']))}] ", style="dim")
            anomaly_text.append(f"{anomaly['type']}: ", style="bold yellow")
            anomaly_text.append(f"{anomaly['details']}\n", style="yellow")
        layout["anomalies"].update(Panel(anomaly_text, title="Anomalies", border_style="yellow"))
    else:
        layout["anomalies"].update(Panel(Text("No anomalies detected.", style="green"), title="Anomalies", border_style="green"))


    panel = Panel(
        layout,
        title=get_gradient_text("System Monitor", "cyan", "magenta"),
        border_style="cyan"
    )
    return Gradient(panel, colors=["cyan", "bright_magenta"])


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
