# Core Skills Package
# Story 2.1: Skill Library Persistence Pipeline

from pathlib import Path

SKILLS_ROOT = Path(__file__).parent
SKILL_CATEGORIES = ["filesystem", "web", "data", "analysis", "generation"]

def get_skill_path(category: str, skill_name: str) -> Path:
    """Get the path for a skill file."""
    return SKILLS_ROOT / category / f"{skill_name}.py"

def list_skill_categories() -> list:
    """List available skill categories."""
    return [d.name for d in SKILLS_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")]
