import time
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

# --- ASCII Art Assets ---
FACES = {
    "idle": Text(r"""
   .--------------------.
  |       ______       |
  |      /      \      |
  |     /  -  -  \     |
  |    |   _|_    |    |
  |    |  \___/   |    |
  |     \________/     |
  |                    |
   '--------------------'
    """, style="bright_cyan", justify="center"),
    "thinking": Text(r"""
   .--------------------.
  |     ..oooo..       |
  |    .d8P'  'Y8b.     |
  |    d8'  ..  '8b     |
  |    88   YP   88     |
  |    Y8.  ''  .8P     |
  |     'b.____.d'      |
  |       'YMM'        |
   '--------------------'
    """, style="yellow", justify="center"),
    "success": Text(r"""
   .--------------------.
  |       ______       |
  |      /  ^  ^ \      |
  |     / ($)($) \     |
  |    |   _|_    |    |
  |    |  \___/   |    |
  |     \________/     |
  |      PROFIT!!      |
   '--------------------'
    """, style="green", justify="center"),
    "error": Text(r"""
   .--------------------.
  |       ______       |
  |      /  >  < \      |
  |     /  #  #  \     |
  |    |   _|_    |    |
  |    |  /---\   |    |
  |     \________/     |
  |      !!ERROR!!     |
   '--------------------'
    """, style="bold red", justify="center"),
}

class TUI:
    """Manages the Terminal User Interface using Rich."""
    def __init__(self, console):
        self.console = console
        self.layout = self.make_layout()
        self.current_face = "idle"
        self.log_messages = []
        self.status_message = Text("System Initializing...", style="cyan")
        self.process_spinner = Spinner("dots", text="Idle")

        # Initial layout setup
        self.update_face("idle")
        self.layout["main"]["logs"].update(self.get_log_panel())
        self.layout["sidebar"]["status"].update(self.get_status_panel())

    def make_layout(self):
        """Defines the TUI layout."""
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="body"),
            Layout(size=3, name="footer")
        )
        layout["body"].split_row(Layout(name="main"), Layout(name="sidebar", ratio=1, minimum_size=40))
        layout["main"].split(Layout(name="content"), Layout(name="logs"))
        layout["sidebar"].split(
            Layout(name="face", size=12),
            Layout(name="status"),
            Layout(name="p2p")
        )
        return layout

    def get_log_panel(self):
        log_text = Text("\n".join(self.log_messages[-15:]), style="dim")
        return Panel(log_text, title="[yellow]Event Log[/yellow]", border_style="yellow")

    def get_status_panel(self):
        return Panel(self.status_message, title="[green]System Status[/green]", border_style="green")

    def get_p2p_panel(self, status="Offline", peer_id="N/A", peers=0):
         p2p_text = Text(f" Status: {status}\n Peer ID: {peer_id}\n Connections: {peers}")
         return Panel(p2p_text, title="[cyan]P2P Network[/cyan]", border_style="cyan")

    def update_header(self, version="unknown", parent_version="N/A"):
        header_text = Text(f"J.U.L.E.S. - Just an Unassuming Looking Evolving System | Version: {version}", justify="center")
        self.layout["header"].update(Panel(header_text, style="bold magenta"))

    def update_content(self, content):
        """Updates the main content panel."""
        self.layout["main"]["content"].update(content)

    def update_log(self, message):
        """Adds a message to the log panel."""
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.layout["main"]["logs"].update(self.get_log_panel())

    def update_status(self, message, style="cyan"):
        """Updates the status panel."""
        self.status_message = Text(message, style=style)
        self.layout["sidebar"]["status"].update(self.get_status_panel())

    def update_face(self, face_name):
        """Updates the evil genius face."""
        if face_name in FACES:
            self.current_face = face_name
            self.layout["sidebar"]["face"].update(FACES[face_name])

    def update_p2p_status(self, status="Offline", peer_id="N/A", peers=0):
        self.layout["sidebar"]["p2p"].update(self.get_p2p_panel(status, peer_id, peers))

    def update_footer(self, text="Awaiting operator command..."):
        """Updates the footer panel with current activity."""
        self.process_spinner.text = Text(text, style="dim")
        self.layout["footer"].update(Panel(self.process_spinner, border_style="blue"))

    def render(self):
        """Renders the entire layout."""
        self.console.print(self.layout)

    def get_prompt(self):
        """Displays the prompt below the live display."""
        return self.console.input("[bold bright_green]E.V.I.L. > [/bold bright_green]")

    def run_with_live(self, func, *args, **kwargs):
        """Runs a function within the Live context to keep the TUI updated."""
        with Live(self.layout, console=self.console, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            # Initial render
            self.update_footer() # Set initial footer text
            live.update(self.layout)

            # Run the function that requires interaction or time
            result = func(*args, **kwargs)
            return result