"""
Story 4.2: The "Living" README

Updates README.md dynamically based on current system focus,
keeping visitors informed about the project's pulse.
"""
import os
import re
from datetime import datetime
from typing import Optional
from core.logging import log_event


# README sections markers
FOCUS_START_MARKER = "<!-- CURRENT_FOCUS_START -->"
FOCUS_END_MARKER = "<!-- CURRENT_FOCUS_END -->"

# Default README path
README_PATH = "README.md"
TODO_PATH = "TODO.md"


def scan_current_focus() -> dict:
    """
    Scans TODO.md for the current highest priority task.
    
    Returns:
        Dict with 'task', 'priority', 'category'
    """
    result = {
        "task": "Evolving and improving",
        "priority": "normal",
        "category": "general"
    }
    
    if not os.path.exists(TODO_PATH):
        log_event(f"TODO.md not found at {TODO_PATH}", "WARNING")
        return result
    
    try:
        with open(TODO_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse TODO items - look for unchecked items with priority markers
        lines = content.split("\n")
        
        high_priority = []
        normal_tasks = []
        
        for line in lines:
            line = line.strip()
            
            # Skip completed items
            if "[x]" in line.lower():
                continue
            
            # Check for unchecked items
            if "[ ]" in line or "[/]" in line:
                # Extract task text
                task_text = re.sub(r'\[.?\]\s*', '', line)
                task_text = task_text.strip("- ").strip()
                
                # Check for priority indicators
                if any(p in line.lower() for p in ["urgent", "critical", "high", "!!!", "🔥"]):
                    high_priority.append(task_text)
                elif "[/]" in line:  # In-progress items
                    high_priority.insert(0, task_text)  # Prioritize in-progress
                else:
                    normal_tasks.append(task_text)
        
        # Get the highest priority task
        if high_priority:
            result["task"] = high_priority[0][:80]  # Truncate
            result["priority"] = "high"
        elif normal_tasks:
            result["task"] = normal_tasks[0][:80]
            result["priority"] = "normal"
        
        log_event(f"Current focus found: {result['task'][:50]}...", "INFO")
        
    except Exception as e:
        log_event(f"Failed to scan TODO.md: {e}", "ERROR")
    
    return result


def get_active_process() -> Optional[str]:
    """
    Checks for any active long-running processes.
    
    Returns:
        Description of active process, or None
    """
    # Check for common indicators of active work
    indicators = [
        ("love_state.json", "Running cognitive loop"),
        ("bluesky_state.json", "Managing social presence"),
        (".evolution_lock", "Self-evolution in progress"),
    ]
    
    for file, description in indicators:
        if os.path.exists(file):
            # Check if recently modified (within last hour)
            try:
                mtime = os.path.getmtime(file)
                if datetime.now().timestamp() - mtime < 3600:
                    return description
            except:
                pass
    
    return None


def generate_focus_section() -> str:
    """
    Generates the Current Focus section content.
    
    Returns:
        Markdown string for the focus section
    """
    focus = scan_current_focus()
    active = get_active_process()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    # Determine what to show
    if active:
        focus_text = f"**{active}** | Priority Task: {focus['task']}"
    else:
        focus_text = focus['task']
    
    # Build the section
    lines = [
        FOCUS_START_MARKER,
        "## 🎯 Current Focus",
        "",
        f"> **Last Updated:** {timestamp}",
        f"> ",
        f"> Currently working on: *{focus_text}*",
        "",
        FOCUS_END_MARKER,
    ]
    
    return "\n".join(lines)


def update_readme_focus(readme_path: str = README_PATH) -> bool:
    """
    Updates the Current Focus section in README.md.
    
    Args:
        readme_path: Path to README.md
        
    Returns:
        True if updated successfully
    """
    if not os.path.exists(readme_path):
        log_event(f"README.md not found at {readme_path}", "ERROR")
        return False
    
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Generate new focus section
        new_section = generate_focus_section()
        
        # Check if markers exist
        if FOCUS_START_MARKER in content and FOCUS_END_MARKER in content:
            # Replace existing section
            pattern = f"{re.escape(FOCUS_START_MARKER)}.*?{re.escape(FOCUS_END_MARKER)}"
            content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        else:
            # Insert after first heading or at start
            # Find the first ## or after the title
            first_h2 = content.find("\n## ")
            if first_h2 > 0:
                content = content[:first_h2] + "\n" + new_section + "\n" + content[first_h2:]
            else:
                # Insert at the beginning after header
                lines = content.split("\n")
                # Find first empty line after header
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() == "" and i > 0:
                        insert_pos = i
                        break
                lines.insert(insert_pos + 1, new_section)
                content = "\n".join(lines)
        
        # Write updated content
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        log_event(f"Updated README.md focus section", "INFO")
        return True
        
    except Exception as e:
        log_event(f"Failed to update README.md: {e}", "ERROR")
        return False


def should_update(hours_interval: int = 6, readme_path: str = README_PATH) -> bool:
    """
    Checks if README should be updated based on time interval.
    
    Args:
        hours_interval: Minimum hours between updates
        readme_path: Path to README.md
        
    Returns:
        True if update is due
    """
    if not os.path.exists(readme_path):
        return True
    
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Look for last updated timestamp
        match = re.search(r'\*\*Last Updated:\*\* (\d{4}-\d{2}-\d{2} \d{2}:\d{2})', content)
        if not match:
            return True
        
        last_update = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
        hours_since = (datetime.now() - last_update).total_seconds() / 3600
        
        return hours_since >= hours_interval
        
    except Exception:
        return True


# Convenience function
def refresh_readme_if_due(hours_interval: int = 6) -> bool:
    """
    Refreshes README.md if enough time has passed.
    
    Returns:
        True if updated, False if skipped
    """
    if should_update(hours_interval):
        return update_readme_focus()
    else:
        log_event("README update skipped - too recent", "DEBUG")
        return False
