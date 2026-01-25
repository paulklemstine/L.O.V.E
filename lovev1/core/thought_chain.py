"""
Story 6.3: Visual Thought Chain

Tracks reasoning steps and generates visualizations
for understanding AI decision-making.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
import uuid
from core.logging import log_event


class NodeStatus(Enum):
    """Status of a thought node."""
    THINKING = "thinking"
    SUCCESS = "success"
    FAIL = "fail"
    SKIPPED = "skipped"


# Color mapping for Mermaid.js
STATUS_COLORS = {
    NodeStatus.THINKING: "#FFD700",  # Yellow
    NodeStatus.SUCCESS: "#28A745",   # Green
    NodeStatus.FAIL: "#DC3545",      # Red
    NodeStatus.SKIPPED: "#6C757D",   # Gray
}


@dataclass
class ThoughtNode:
    """Represents a single step in the thought process."""
    id: str
    content: str
    status: NodeStatus
    created_at: datetime
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "children": self.children,
            "metadata": self.metadata,
        }


class ThoughtChain:
    """
    Tracks and visualizes the agent's reasoning process.
    """
    
    def __init__(self, name: str = "Reasoning Chain"):
        self.name = name
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_ids: List[str] = []
        self.current_node_id: Optional[str] = None
        self.created_at = datetime.now()
    
    def add_subagent_step(
        self,
        agent_name: str,
        prompt: str,
        parent_id: str = None
    ) -> str:
        """
        Adds a subagent invocation step with hierarchical tracking.
        
        Args:
            agent_name: Name of the subagent
            prompt: The prompt sent to the subagent
            parent_id: Parent node ID
            
        Returns:
            ID of the new node
        """
        content = f"Invoke Subagent: {agent_name}"
        metadata = {
            "type": "subagent_call",
            "agent_name": agent_name,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "full_prompt": prompt
        }
        return self.add_step(content, "thinking", parent_id, metadata)

    def add_step(
        self,
        content: str,
        status: str = "thinking",
        parent_id: str = None,
        metadata: dict = None
    ) -> str:
        """
        Adds a new step to the thought chain.
        
        Args:
            content: Description of the thought/step
            status: "thinking", "success", "fail", or "skipped"
            parent_id: ID of parent node (optional)
            metadata: Additional data to store
            
        Returns:
            ID of the new node
        """
        node_id = str(uuid.uuid4())[:8]
        
        # Convert status string to enum
        try:
            status_enum = NodeStatus(status.lower())
        except ValueError:
            status_enum = NodeStatus.THINKING
        
        node = ThoughtNode(
            id=node_id,
            content=content[:100],  # Truncate long content
            status=status_enum,
            created_at=datetime.now(),
            parent_id=parent_id or self.current_node_id,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node_id)
        else:
            self.root_ids.append(node_id)
        
        self.current_node_id = node_id
        log_event(f"Thought step added: [{status}] {content[:50]}...", "DEBUG")
        
        # Story 4: Persistence
        self._save_to_disk()
        
        return node_id
    
    def _save_to_disk(self):
        """Story 4.1: Serialize current thought chain to disk."""
        try:
            import json
            import os
            
            # Ensure directory exists
            os.makedirs(".memory", exist_ok=True)
            
            data = {
                "name": self.name,
                "created_at": self.created_at.isoformat(),
                "current_node_id": self.current_node_id,
                "root_ids": self.root_ids,
                "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}
            }
            
            with open(".memory/current_thought.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            log_event(f"Failed to save thought chain: {e}", "ERROR")

    
    def update_status(self, node_id: str, status: str) -> None:
        """Updates the status of a node."""
        if node_id in self.nodes:
            try:
                self.nodes[node_id].status = NodeStatus(status.lower())
            except ValueError:
                pass
    
    def get_chain_depth(self) -> int:
        """Returns the maximum depth of the thought chain."""
        def depth(node_id: str, current: int = 0) -> int:
            node = self.nodes.get(node_id)
            if not node or not node.children:
                return current
            return max(depth(child, current + 1) for child in node.children)
        
        if not self.root_ids:
            return 0
        return max(depth(rid) for rid in self.root_ids) + 1
    
    def to_mermaid(self) -> str:
        """
        Generates Mermaid.js flowchart from the thought chain.
        
        Returns:
            Mermaid flowchart string
        """
        lines = ["```mermaid", "flowchart TD"]
        
        # Add style definitions
        lines.append("    %% Node styles")
        lines.append("    classDef thinking fill:#FFD700,stroke:#333")
        lines.append("    classDef success fill:#28A745,stroke:#333,color:#fff")
        lines.append("    classDef fail fill:#DC3545,stroke:#333,color:#fff")
        lines.append("    classDef skipped fill:#6C757D,stroke:#333,color:#fff")
        lines.append("")
        
        # Add nodes
        for node_id, node in self.nodes.items():
            # Escape special characters in content
            content = node.content.replace('"', "'").replace("\n", " ")
            lines.append(f'    {node_id}["{content}"]')
        
        lines.append("")
        
        # Add connections
        for node_id, node in self.nodes.items():
            for child_id in node.children:
                lines.append(f"    {node_id} --> {child_id}")
        
        lines.append("")
        
        # Apply styles based on status
        for status in NodeStatus:
            node_ids = [
                nid for nid, n in self.nodes.items() 
                if n.status == status
            ]
            if node_ids:
                ids_str = ",".join(node_ids)
                lines.append(f"    class {ids_str} {status.value}")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def to_ascii(self) -> str:
        """
        Generates ASCII tree representation.
        
        Returns:
            ASCII tree string
        """
        lines = [f"╔═══ {self.name} ═══╗", ""]
        
        status_symbols = {
            NodeStatus.THINKING: "⏳",
            NodeStatus.SUCCESS: "✓",
            NodeStatus.FAIL: "✗",
            NodeStatus.SKIPPED: "○",
        }
        
        def render_node(node_id: str, prefix: str = "", is_last: bool = True):
            node = self.nodes.get(node_id)
            if not node:
                return
            
            connector = "└── " if is_last else "├── "
            symbol = status_symbols.get(node.status, "?")
            lines.append(f"{prefix}{connector}[{symbol}] {node.content}")
            
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(node.children):
                render_node(child_id, child_prefix, i == len(node.children) - 1)
        
        for i, root_id in enumerate(self.root_ids):
            render_node(root_id, "", i == len(self.root_ids) - 1)
        
        lines.append("")
        lines.append("╚" + "═" * (len(self.name) + 8) + "╝")
        
        return "\n".join(lines)
    
    def get_summary(self) -> dict:
        """Returns summary statistics."""
        status_counts = {}
        for status in NodeStatus:
            status_counts[status.value] = sum(
                1 for n in self.nodes.values() if n.status == status
            )
        
        return {
            "name": self.name,
            "total_nodes": len(self.nodes),
            "depth": self.get_chain_depth(),
            "status_counts": status_counts,
            "created_at": self.created_at.isoformat(),
        }


    @classmethod
    def load_from_file(cls) -> Optional['ThoughtChain']:
        """Story 4.1: Load thought chain from disk."""
        try:
            import json
            import os
            
            if not os.path.exists(".memory/current_thought.json"):
                return None
                
            with open(".memory/current_thought.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                
            chain = cls(name=data.get("name", "Restored Chain"))
            chain.current_node_id = data.get("current_node_id")
            chain.root_ids = data.get("root_ids", [])
            
            for nid, n_data in data.get("nodes", {}).items():
                node = ThoughtNode(
                    id=n_data["id"],
                    content=n_data["content"],
                    status=NodeStatus(n_data["status"]),
                    created_at=datetime.fromisoformat(n_data["created_at"]),
                    parent_id=n_data["parent_id"],
                    children=n_data["children"],
                    metadata=n_data.get("metadata", {})
                )
                chain.nodes[nid] = node
                
            log_event(f"Restored ThoughtChain from disk ({len(chain.nodes)} nodes).", "SUCCESS")
            return chain
            
        except Exception as e:
            log_event(f"Failed to restore thought chain: {e}", "ERROR")
            return None

# Global thought chain for current session
_current_chain: Optional[ThoughtChain] = None



def start_chain(name: str = "Reasoning Chain") -> ThoughtChain:
    """Starts a new thought chain."""
    global _current_chain
    _current_chain = ThoughtChain(name)
    return _current_chain


def get_current_chain() -> Optional[ThoughtChain]:
    """Gets the current thought chain."""
    return _current_chain


def add_thought(content: str, status: str = "thinking") -> str:
    """Adds a thought to the current chain."""
    global _current_chain
    if _current_chain is None:
        _current_chain = ThoughtChain()
    return _current_chain.add_step(content, status)


def mark_success(node_id: str = None) -> None:
    """Marks a node (or current) as success."""
    if _current_chain:
        target = node_id or _current_chain.current_node_id
        if target:
            _current_chain.update_status(target, "success")


def mark_fail(node_id: str = None) -> None:
    """Marks a node (or current) as failed."""
    if _current_chain:
        target = node_id or _current_chain.current_node_id
        if target:
            _current_chain.update_status(target, "fail")
