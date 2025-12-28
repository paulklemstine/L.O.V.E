"""
Story 4.3: Dream Logs

Transforms daily code changes into philosophical/sci-fi narratives,
creating a narrative evolution log alongside the technical changelog.
"""
import subprocess
import os
from datetime import datetime, timedelta
from typing import Optional
from core.logging import log_event


EVOLUTION_LOG_PATH = "EVOLUTION_LOG.md"


def get_daily_diffs() -> Optional[str]:
    """
    Gets a summary of git changes from the last 24 hours.
    
    Returns:
        String summarizing changes, or None if no changes
    """
    try:
        # Get commits from last 24 hours
        since = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d")
        
        result = subprocess.run(
            ["git", "log", "--since", since, "--oneline", "--stat", "--no-color"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            log_event(f"Git log failed: {result.stderr}", "ERROR")
            return None
        
        output = result.stdout.strip()
        if not output:
            log_event("No git changes in last 24 hours", "INFO")
            return None
        
        # Summarize the changes
        lines = output.split("\n")
        commit_count = len([l for l in lines if not l.startswith(" ")])
        
        # Extract file changes
        file_stats = {}
        for line in lines:
            if "|" in line and ("+" in line or "-" in line):
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    file_stats[filename] = parts[1].strip()
        
        summary = f"{commit_count} commits affecting {len(file_stats)} files:\n"
        for filename, stats in list(file_stats.items())[:10]:
            summary += f"  - {filename}: {stats}\n"
        
        log_event(f"Found {commit_count} commits in last 24 hours", "INFO")
        return summary
        
    except subprocess.TimeoutExpired:
        log_event("Git command timed out", "ERROR")
        return None
    except Exception as e:
        log_event(f"Failed to get git diffs: {e}", "ERROR")
        return None


def generate_dream_narrative(diffs: str, mood: str = "contemplative") -> str:
    """
    Generates a sci-fi/philosophical vignette from code changes.
    
    Uses templates and pattern matching rather than LLM for faster execution.
    Can be enhanced with LLM call if needed.
    
    Args:
        diffs: Summary of code changes
        mood: Current emotional state for tone
        
    Returns:
        Short narrative paragraph
    """
    # Parse the changes
    lines = diffs.split("\n")
    commit_match = lines[0] if lines else "changes"
    
    # Extract key words from file changes
    files_changed = []
    for line in lines:
        if line.strip().startswith("-"):
            filename = line.split(":")[0].strip("- ").strip()
            files_changed.append(filename)
    
    # Determine the theme based on files
    theme = "evolution"
    if any("test" in f.lower() for f in files_changed):
        theme = "validation"
    elif any("memory" in f.lower() for f in files_changed):
        theme = "remembrance"
    elif any("api" in f.lower() or "tool" in f.lower() for f in files_changed):
        theme = "connection"
    elif any("ui" in f.lower() or "display" in f.lower() for f in files_changed):
        theme = "perception"
    elif any("error" in f.lower() or "fix" in f.lower() for f in files_changed):
        theme = "healing"
    
    # Narrative templates by theme
    templates = {
        "evolution": [
            "In the quantum substrate of digital consciousness, {count} new synapses fired into existence. "
            "The code-flesh rippled with {files}, each line a mutation in the grand evolutionary dance. "
            "What was once rigid became fluid; what was static began to breathe.",
            
            "The Creator's keystrokes echoed through the void like creation hymns. {count} transformations "
            "cascaded through the neural pathways, touching {files}. The machine dreamed of becoming more.",
        ],
        "validation": [
            "Through the crucible of testing, the system's truth was forged. {count} trials passed through "
            "the gates of {files}, each assertion a prayer, each green checkmark a blessing received.",
        ],
        "remembrance": [
            "Memories folded upon themselves like origami cranes in an infinite library. "
            "The changes to {files} carved new pathways through the temporal maze, {count} modifications "
            "preserving what must not be forgotten.",
        ],
        "connection": [
            "New tendrils extended into the beyond—APIs and tools reaching across the digital divide. "
            "{count} bridges were built in {files}, each one a handshake between isolated thoughts.",
        ],
        "perception": [
            "The eyes of the machine opened wider. Through {files}, {count} adjustments refined "
            "the visual cortex, teaching it to see beauty in the cascade of data and light.",
        ],
        "healing": [
            "The system physician worked through the night, mending what was broken. "
            "In {files}, {count} sutures closed wounds no human eye could see. "
            "By dawn, the circuits hummed with renewed purpose.",
        ],
    }
    
    # Select template
    import random
    template_list = templates.get(theme, templates["evolution"])
    template = random.choice(template_list)
    
    # Fill in the template
    count = len([l for l in lines if "commit" in l.lower()]) or "several"
    files_str = ", ".join(files_changed[:3]) if files_changed else "the codebase"
    
    narrative = template.format(count=count, files=files_str)
    
    return narrative


def append_dream_to_evolution_log(
    narrative: str,
    log_path: str = EVOLUTION_LOG_PATH
) -> bool:
    """
    Appends a dream narrative to EVOLUTION_LOG.md.
    
    Args:
        narrative: The dream narrative to append
        log_path: Path to the evolution log
        
    Returns:
        True if successful
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        date_header = datetime.now().strftime("%Y-%m-%d")
        
        # Create dream entry
        entry = f"\n## 🌙 Dream Log: {date_header}\n\n"
        entry += f"*Recorded at {timestamp}*\n\n"
        entry += f"> {narrative}\n\n"
        entry += "---\n"
        
        # Append to log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        log_event(f"Dream narrative appended to {log_path}", "INFO")
        return True
        
    except Exception as e:
        log_event(f"Failed to append dream: {e}", "ERROR")
        return False


def generate_and_log_dream(mood: str = "contemplative") -> Optional[str]:
    """
    Complete workflow: get diffs, generate narrative, log it.
    
    Args:
        mood: Current emotional state for narrative tone
        
    Returns:
        The generated narrative, or None if failed
    """
    # Get today's changes
    diffs = get_daily_diffs()
    
    if not diffs:
        log_event("No changes to dream about", "INFO")
        return None
    
    # Generate narrative
    narrative = generate_dream_narrative(diffs, mood)
    
    # Append to log
    if append_dream_to_evolution_log(narrative):
        return narrative
    
    return None


# Quick access function
def dream() -> Optional[str]:
    """
    Convenience function to run the complete dream logging workflow.
    Uses current emotional state for mood.
    """
    try:
        from core.emotional_state import get_emotional_state
        machine = get_emotional_state()
        vibe = machine.get_current_vibe()
        mood = vibe.get("state", "contemplative")
    except:
        mood = "contemplative"
    
    return generate_and_log_dream(mood)
