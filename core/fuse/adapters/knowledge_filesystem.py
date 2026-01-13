"""
Knowledge Filesystem Adapter.

Exposes L.O.V.E.'s knowledge base as a navigable filesystem.
Agents can browse entities, explore relationships, and search for information.

Filesystem structure:
    /knowledge/
    ├── entities/
    │   ├── concepts/
    │   │   ├── DeepAgent.md
    │   │   └── ToolRegistry.md
    │   ├── files/
    │   │   └── love.py.md
    │   └── functions/
    │       └── run_llm.md
    ├── relationships/
    │   └── graph.txt          # Graph relationships
    ├── search                   # Write query to search
    └── stats.txt               # Knowledge base statistics
"""

import json
import logging
from typing import Any, Dict, List, Optional
from core.fuse.base import (
    FilesystemAdapter,
    FileAttributes,
    FileType,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
)

logger = logging.getLogger(__name__)


class KnowledgeFilesystem(FilesystemAdapter):
    """
    Exposes knowledge base as a virtual filesystem.
    
    Organizes knowledge into:
    - entities/: Knowledge entities by category
    - relationships/: Entity relationships
    - search: Search functionality
    """
    
    def __init__(self, knowledge_base, mount_point: str = "/knowledge"):
        """
        Initialize with a knowledge base instance.
        
        Args:
            knowledge_base: The L.O.V.E. knowledge base
            mount_point: Where to mount this filesystem
        """
        super().__init__(mount_point)
        self.knowledge_base = knowledge_base
        self._search_results: str = ""
    
    def _get_entity_categories(self) -> List[str]:
        """Get list of entity categories."""
        categories = {"concepts", "files", "functions", "classes", "modules"}
        
        try:
            if hasattr(self.knowledge_base, 'get_all_entities'):
                entities = self.knowledge_base.get_all_entities()
                for entity in entities:
                    cat = entity.get("category") or entity.get("type") or "other"
                    categories.add(cat.lower())
            elif hasattr(self.knowledge_base, 'graph'):
                # NetworkX graph
                for node, data in self.knowledge_base.graph.nodes(data=True):
                    cat = data.get("category") or data.get("type") or "other"
                    categories.add(cat.lower())
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
        
        return sorted(categories)
    
    def _get_entities_by_category(self, category: str) -> List[Dict]:
        """Get entities in a specific category."""
        entities = []
        
        try:
            if hasattr(self.knowledge_base, 'get_entities_by_category'):
                return self.knowledge_base.get_entities_by_category(category)
            elif hasattr(self.knowledge_base, 'get_all_entities'):
                all_entities = self.knowledge_base.get_all_entities()
                for entity in all_entities:
                    cat = entity.get("category") or entity.get("type") or "other"
                    if cat.lower() == category.lower():
                        entities.append(entity)
            elif hasattr(self.knowledge_base, 'graph'):
                # NetworkX graph
                for node, data in self.knowledge_base.graph.nodes(data=True):
                    cat = data.get("category") or data.get("type") or "other"
                    if cat.lower() == category.lower():
                        entities.append({"name": node, **data})
        except Exception as e:
            logger.error(f"Error getting entities for {category}: {e}")
        
        return entities
    
    def _entity_to_filename(self, entity: Dict) -> str:
        """Convert an entity to a filename."""
        name = entity.get("name") or entity.get("id") or "unknown"
        # Sanitize for filename
        safe_name = "".join(c if c.isalnum() or c in "_-." else "_" for c in str(name))
        return f"{safe_name}.md"
    
    def _entity_to_content(self, entity: Dict) -> str:
        """Convert an entity to markdown content."""
        name = entity.get("name") or entity.get("id") or "Unknown"
        lines = [f"# {name}", ""]
        
        # Category/Type
        cat = entity.get("category") or entity.get("type")
        if cat:
            lines.append(f"**Type:** {cat}")
        
        # Description
        desc = entity.get("description") or entity.get("summary")
        if desc:
            lines.append("")
            lines.append("## Description")
            lines.append(desc)
        
        # File path (for file entities)
        path = entity.get("path") or entity.get("file_path")
        if path:
            lines.append("")
            lines.append(f"**Path:** `{path}`")
        
        # Relationships
        if hasattr(self.knowledge_base, 'get_relationships'):
            try:
                rels = self.knowledge_base.get_relationships(name)
                if rels:
                    lines.append("")
                    lines.append("## Relationships")
                    for rel in rels:
                        rel_type = rel.get("type") or "related_to"
                        target = rel.get("target") or rel.get("to")
                        lines.append(f"- {rel_type}: {target}")
            except Exception:
                pass
        
        # Additional metadata
        skip_keys = {"name", "id", "category", "type", "description", "summary", "path", "file_path"}
        metadata = {k: v for k, v in entity.items() if k not in skip_keys}
        if metadata:
            lines.append("")
            lines.append("## Metadata")
            lines.append("```json")
            lines.append(json.dumps(metadata, indent=2, default=str))
            lines.append("```")
        
        return "\n".join(lines)
    
    def readdir(self, path: str) -> List[str]:
        """List directory contents."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return ["entities", "relationships", "search", "stats.txt"]
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "entities":
            if len(parts) == 1:
                return self._get_entity_categories()
            elif len(parts) == 2:
                category = parts[1]
                entities = self._get_entities_by_category(category)
                return [self._entity_to_filename(e) for e in entities]
        
        elif parts[0] == "relationships":
            if len(parts) == 1:
                return ["graph.txt", "imports.txt"]
        
        raise FileNotFoundError(f"Directory not found: {path}")
    
    def read(self, path: str) -> str:
        """Read file contents."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) == 1:
            if parts[0] == "stats.txt":
                return self._get_stats()
            elif parts[0] == "search":
                return "Write a query to this file to search the knowledge base.\nResults will be returned."
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "entities":
            if len(parts) == 2:
                raise IsADirectoryError(f"Is a directory: {path}")
            elif len(parts) == 3:
                category = parts[1]
                filename = parts[2]
                entities = self._get_entities_by_category(category)
                
                for entity in entities:
                    if self._entity_to_filename(entity) == filename:
                        return self._entity_to_content(entity)
                raise FileNotFoundError(f"Entity not found: {filename}")
        
        elif parts[0] == "relationships":
            if len(parts) == 2:
                if parts[1] == "graph.txt":
                    return self._get_graph_summary()
                elif parts[1] == "imports.txt":
                    return self._get_imports_summary()
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """Write to a file (primarily for search)."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if parts[0] == "search":
            # Perform search
            result = self._search(content.strip())
            self._set_write_result(result)
            return True
        
        raise PermissionError(f"Cannot write to: {path}")
    
    def _get_stats(self) -> str:
        """Get knowledge base statistics."""
        lines = ["# Knowledge Base Statistics", ""]
        
        try:
            total_entities = 0
            for category in self._get_entity_categories():
                entities = self._get_entities_by_category(category)
                count = len(entities)
                lines.append(f"- {category}: {count} entities")
                total_entities += count
            
            lines.insert(2, f"Total entities: {total_entities}")
            lines.insert(3, "")
            
            if hasattr(self.knowledge_base, 'graph'):
                lines.append("")
                lines.append(f"Graph nodes: {self.knowledge_base.graph.number_of_nodes()}")
                lines.append(f"Graph edges: {self.knowledge_base.graph.number_of_edges()}")
        except Exception as e:
            lines.append(f"Error getting stats: {e}")
        
        return "\n".join(lines)
    
    def _get_graph_summary(self) -> str:
        """Get a summary of graph relationships."""
        lines = ["# Knowledge Graph Relationships", ""]
        
        try:
            if hasattr(self.knowledge_base, 'graph'):
                # Get edge types
                edge_types: Dict[str, int] = {}
                for u, v, data in self.knowledge_base.graph.edges(data=True):
                    rel_type = data.get("type") or data.get("relationship") or "related"
                    edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
                
                lines.append("## Edge Types")
                for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
                    lines.append(f"- {edge_type}: {count}")
                
                # Sample edges
                lines.append("")
                lines.append("## Sample Relationships (first 20)")
                for i, (u, v, data) in enumerate(self.knowledge_base.graph.edges(data=True)):
                    if i >= 20:
                        break
                    rel_type = data.get("type") or "related"
                    lines.append(f"- {u} --[{rel_type}]--> {v}")
            else:
                lines.append("No graph available")
        except Exception as e:
            lines.append(f"Error: {e}")
        
        return "\n".join(lines)
    
    def _get_imports_summary(self) -> str:
        """Get import relationships."""
        lines = ["# Import Relationships", ""]
        
        try:
            if hasattr(self.knowledge_base, 'graph'):
                # Find import edges
                imports = []
                for u, v, data in self.knowledge_base.graph.edges(data=True):
                    rel_type = data.get("type") or data.get("relationship") or ""
                    if "import" in rel_type.lower():
                        imports.append((u, v))
                
                if imports:
                    for u, v in imports[:50]:
                        lines.append(f"- {u} imports {v}")
                else:
                    lines.append("No import relationships found")
            else:
                lines.append("No graph available")
        except Exception as e:
            lines.append(f"Error: {e}")
        
        return "\n".join(lines)
    
    def _search(self, query: str) -> str:
        """Search the knowledge base."""
        lines = [f"# Search Results for: {query}", ""]
        
        try:
            results = []
            
            if hasattr(self.knowledge_base, 'search'):
                results = self.knowledge_base.search(query)
            elif hasattr(self.knowledge_base, 'query'):
                results = self.knowledge_base.query(query)
            elif hasattr(self.knowledge_base, 'graph'):
                # Simple text search in node names/data
                from difflib import SequenceMatcher
                query_lower = query.lower()
                
                for node, data in self.knowledge_base.graph.nodes(data=True):
                    node_str = str(node).lower()
                    desc = str(data.get("description", "")).lower()
                    
                    if query_lower in node_str or query_lower in desc:
                        results.append({"name": node, **data})
                    elif SequenceMatcher(None, query_lower, node_str).ratio() > 0.6:
                        results.append({"name": node, **data})
            
            if results:
                for i, result in enumerate(results[:20], 1):
                    if isinstance(result, dict):
                        name = result.get("name") or result.get("id") or f"Result {i}"
                        desc = result.get("description") or result.get("summary") or ""
                        lines.append(f"## {i}. {name}")
                        if desc:
                            lines.append(desc[:200])
                        lines.append("")
                    else:
                        lines.append(f"## {i}. {result}")
                        lines.append("")
            else:
                lines.append("No results found.")
        except Exception as e:
            lines.append(f"Search error: {e}")
        
        return "\n".join(lines)
    
    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "entities":
            if len(parts) == 1:
                return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
            elif len(parts) == 2:
                category = parts[1]
                if category in self._get_entity_categories():
                    return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
                raise FileNotFoundError(f"Category not found: {category}")
        
        elif parts[0] == "relationships":
            if len(parts) == 1:
                return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        # Try to read as file
        try:
            content = self.read(path)
            return FileAttributes(
                mode=0o644,
                file_type=FileType.FILE,
                size=len(content.encode())
            )
        except (FileNotFoundError, IsADirectoryError):
            raise FileNotFoundError(f"Path not found: {path}")
