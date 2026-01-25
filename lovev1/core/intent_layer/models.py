from dataclasses import dataclass, field
from typing import List, Optional

INTENT_FILENAME = "AGENTS.md"

@dataclass
class IntentNode:
    """
    Represents a single node in the Intent Layer hierarchy.
    """
    path: str
    content: str
    summary: Optional[str] = None
    parent: Optional['IntentNode'] = None
    downlinks: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"<IntentNode path='{self.path}'>"
