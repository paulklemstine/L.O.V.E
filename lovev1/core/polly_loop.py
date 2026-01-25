
import asyncio
import random
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.console import Group
from rich_gradient.gradient import Gradient

from core.polly import PollyOptimizer
from core.prompt_registry import get_prompt_registry
from core.logging import log_event
from ui_utils import get_gradient_text, RAVE_COLORS, get_rave_emoji

class PollyOptimizationLoop:
    def __init__(self, ui_queue=None, interval_seconds: int = 3600): # Default to 1 hour
        self.interval = interval_seconds
        self.optimizer = PollyOptimizer()
        self.running = False
        self.ui_queue = ui_queue

    def _create_polly_panel(self, message: str, status: str = "active"):
        """Creates a vibrant, dopamine-inducing panel for Polly."""
        width = 80 # default
        
        # Colors & Emojis
        if status == "success":
            border_color = "bright_green"
            title_emoji = "✨🦜"
            msg_style = "bold bright_green"
        elif status == "optimizing":
            border_color = "bright_cyan"
            title_emoji = "🦜🔍"
            msg_style = "cyan"
        elif status == "fail":
            border_color = "bright_red"
            title_emoji = "🦜⚠️"
            msg_style = "red"
        else:
            border_color = "bright_magenta"
            title_emoji = "🦜"
            msg_style = "white"

        title_text = f"{title_emoji} POLLY EVOLUTION {title_emoji}"
        title = get_gradient_text(title_text, border_color, random.choice(RAVE_COLORS))
        
        # Content with raving bottom bar
        content_items = []
        content_items.append(Align.center(Text(message, style=msg_style)))
        
        # Add a cute bottom bar
        bottom_bar = Text()
        for _ in range(10):
             bottom_bar.append(f"{get_rave_emoji()} ", style=random.choice(RAVE_COLORS))
        content_items.append(Text("\n"))
        content_items.append(Align.center(bottom_bar))

        panel = Panel(
            Group(*content_items),
            title=title,
            border_style=border_color,
            width=width,
            padding=(1, 2)
        )
        return Gradient(panel, colors=[border_color, random.choice(RAVE_COLORS)])

    async def _emit_ui(self, message: str, status: str = "active"):
        if self.ui_queue:
            panel = self._create_polly_panel(message, status)
            self.ui_queue.put(panel)

    async def start(self):
        self.running = True
        log_event("Polly Loop: Starting constant prompt optimization...", "INFO")
        
        while self.running:
            try:
                # 1. Get all prompt keys
                registry = get_prompt_registry()
                # registry._prompts might be empty if not loaded, ensure loaded
                if not registry._prompts:
                    registry._load_prompts()
                
                keys = list(registry._prompts.keys())
                if not keys:
                    log_event("Polly Loop: No prompts found to optimize.", "WARNING")
                    await asyncio.sleep(60)
                    continue

                # 2. Pick a random key
                target_key = random.choice(keys)
                log_event(f"Polly Loop: Selected '{target_key}' for optimization.", "INFO")
                await self._emit_ui(f"Manifesting improvements for: {target_key}...", "optimizing")

                # 3. Optimize
                new_prompt = await self.optimizer.optimize_prompt(target_key)
                
                if new_prompt:
                    # 4. Create PR Workflow (Do NOT update main directly)
                    try:
                        import subprocess
                        import os
                        import time
                        import yaml

                        # Ensure git identity
                        subprocess.run(["git", "config", "user.email", "polly@love.love"], check=False)
                        subprocess.run(["git", "config", "user.name", "Polly Evolution"], check=False)
                        
                        # Get current branch to return to
                        res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
                        original_branch = res.stdout.strip() or "main"

                        # Create new branch
                        timestamp = int(time.time())
                        clean_key = target_key.replace('_', '-')
                        branch_name = f"polly/opt-{clean_key}-{timestamp}"
                        
                        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
                        
                        try:
                            # Manually update prompts.yaml ON THE NEW BRANCH
                            prompts_file = "core/prompts.yaml"
                            if os.path.exists(prompts_file):
                                with open(prompts_file, 'r', encoding='utf-8') as f:
                                    data = yaml.safe_load(f) or {}
                                data[target_key] = new_prompt
                                with open(prompts_file, 'w', encoding='utf-8') as f:
                                    yaml.safe_dump(data, f, indent=2, width=4096, allow_unicode=True, default_flow_style=False)
                                
                                # Add and Commit
                                subprocess.run(["git", "add", prompts_file], check=True, capture_output=True)
                                commit_msg = f"Polly: Proposal to optimize '{target_key}'"
                                subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
                                
                                # Push
                                env_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
                                push_remote = "origin"
                                if env_token:
                                    res = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True)
                                    remote_url = res.stdout.strip()
                                    if remote_url.startswith("https://"):
                                        push_remote = remote_url.replace("https://", f"https://{env_token}@")

                                subprocess.run(["git", "push", push_remote, branch_name], check=True, capture_output=True)
                                
                                log_event(f"Polly Loop: Pushed optimization proposal to branch '{branch_name}'.", "INFO")
                                await self._emit_ui(f"✨ EVOLUTION PROPOSED! ✨\nCheck branch '{branch_name}' for PR.", "success")
                            
                            else:
                                 log_event("Polly Loop: prompts.yaml not found for update.", "ERROR")

                        finally:
                            # ALWAYS switch back to original branch
                            subprocess.run(["git", "checkout", original_branch], check=True, capture_output=True)
                            
                    except Exception as e:
                        log_event(f"Polly Loop: PR workflow failed: {e}", "ERROR")
                        await self._emit_ui(f"Polly tired. PR workflow failed: {e}", "fail")
                        # Try to ensure we are back on main if possible, though 'finally' handles it if checkout succeeded
                        subprocess.run(["git", "checkout", "main"], check=False, capture_output=True)
                else:
                    log_event(f"Polly Loop: No improvement found for '{target_key}' or optimization failed.", "INFO")
                    await self._emit_ui(f"No optimization needed for '{target_key}' right now. Already perfect! 💖", "active")

            except Exception as e:
                log_event(f"Polly Loop: Error in optimization cycle: {e}", "ERROR")
                await self._emit_ui(f"Polly fainted! {e}", "fail")

            # Wait for next cycle
            await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False
