"""
Story 1.3: The Codebase Isomorphism

This module generates a "Map" of the codebase where file distance equals
semantic distance. It visualizes which parts of L.O.V.E.'s "brain" work together.

The visualization allows the AI to understand its own architecture and
debug circular dependencies or unexpected couplings.

Usage:
    from core.reflection.visualize_self_model import (
        build_dependency_graph,
        generate_visualization,
        get_codebase_map
    )
    
    # Build the graph
    graph = build_dependency_graph()
    
    # Generate visualization
    generate_visualization(graph, "self_model.png")
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from core.logging import log_event

# Try to import visualization dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORE_DIR = PROJECT_ROOT / "core"
OUTPUT_DIR = PROJECT_ROOT / "state"


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    path: Path
    name: str
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    lines_of_code: int = 0


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract imports from a Python file."""
    
    def __init__(self):
        self.imports = []
        self.from_imports = []
        self.classes = []
        self.functions = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.from_imports.append(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        if not node.name.startswith('_'):  # Skip private functions
            self.functions.append(node.name)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        if not node.name.startswith('_'):
            self.functions.append(node.name)
        self.generic_visit(node)


def parse_python_file(path: Path) -> Optional[ModuleInfo]:
    """
    Parses a Python file and extracts module information.
    
    Args:
        path: Path to the Python file
        
    Returns:
        ModuleInfo object or None if parsing fails
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Calculate relative module name
        try:
            rel_path = path.relative_to(PROJECT_ROOT)
            module_name = str(rel_path).replace('/', '.').replace('\\', '.')[:-3]
        except ValueError:
            module_name = path.stem
        
        return ModuleInfo(
            path=path,
            name=module_name,
            imports=visitor.imports,
            from_imports=visitor.from_imports,
            classes=visitor.classes,
            functions=visitor.functions,
            lines_of_code=len(content.splitlines()),
        )
        
    except Exception as e:
        log_event(f"Failed to parse {path}: {e}", "DEBUG")
        return None


def build_dependency_graph(root_dir: Path = None) -> 'nx.DiGraph':
    """
    Builds a NetworkX directed graph of Python file dependencies.
    
    Nodes are Python modules, edges are import relationships.
    File distance in the graph approximates semantic distance.
    
    Args:
        root_dir: Root directory to scan (defaults to core/)
        
    Returns:
        NetworkX DiGraph
    """
    if not HAS_NETWORKX:
        log_event("NetworkX not installed. Run: pip install networkx", "ERROR")
        return None
    
    root_dir = root_dir or CORE_DIR
    graph = nx.DiGraph()
    
    # Collect all Python files
    py_files = list(root_dir.glob("**/*.py"))
    modules: Dict[str, ModuleInfo] = {}
    
    for py_file in py_files:
        if '__pycache__' in str(py_file):
            continue
            
        info = parse_python_file(py_file)
        if info:
            modules[info.name] = info
            
            # Add node with attributes
            graph.add_node(
                info.name,
                path=str(info.path),
                classes=len(info.classes),
                functions=len(info.functions),
                loc=info.lines_of_code,
            )
    
    # Add edges for dependencies
    for module_name, info in modules.items():
        for imp in info.imports + info.from_imports:
            # Check if this is an internal import
            if imp.startswith('core.'):
                # Normalize to match our module names
                target = imp
                if target in modules:
                    graph.add_edge(module_name, target)
            elif imp in modules:
                graph.add_edge(module_name, target)
    
    log_event(f"Built dependency graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges", "INFO")
    
    return graph


def find_circular_dependencies(graph: 'nx.DiGraph') -> List[List[str]]:
    """
    Finds circular dependencies in the codebase.
    
    This is useful for the AI to understand and fix architectural issues.
    
    Args:
        graph: The dependency graph
        
    Returns:
        List of cycles (each cycle is a list of module names)
    """
    if not HAS_NETWORKX or graph is None:
        return []
    
    try:
        cycles = list(nx.simple_cycles(graph))
        return cycles
    except Exception:
        return []


def get_module_clusters(graph: 'nx.DiGraph') -> Dict[str, List[str]]:
    """
    Identifies clusters of tightly-coupled modules.
    
    Useful for understanding which parts of the brain work together.
    
    Args:
        graph: The dependency graph
        
    Returns:
        Dict mapping cluster_id to list of module names
    """
    if not HAS_NETWORKX or graph is None:
        return {}
    
    # Convert to undirected for community detection
    undirected = graph.to_undirected()
    
    # Use connected components as a simple clustering
    clusters = {}
    for i, component in enumerate(nx.connected_components(undirected)):
        clusters[f"cluster_{i}"] = list(component)
    
    return clusters


def get_most_connected_modules(graph: 'nx.DiGraph', top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Returns the most highly-connected modules (hubs).
    
    These are critical architectural points in the system.
    
    Args:
        graph: The dependency graph
        top_n: Number of modules to return
        
    Returns:
        List of (module_name, total_connections) sorted by connections
    """
    if not HAS_NETWORKX or graph is None:
        return []
    
    connections = []
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        connections.append((node, in_degree + out_degree))
    
    connections.sort(key=lambda x: x[1], reverse=True)
    return connections[:top_n]


def generate_visualization(
    graph: 'nx.DiGraph',
    output_path: str = None,
    title: str = "L.O.V.E. Codebase Architecture"
) -> Optional[str]:
    """
    Generates a PNG visualization of the dependency graph.
    
    Args:
        graph: The dependency graph
        output_path: Path to save the image (defaults to state/codebase_map.png)
        title: Title for the visualization
        
    Returns:
        Path to the generated image, or None if failed
    """
    if not HAS_NETWORKX or not HAS_MATPLOTLIB:
        log_event("NetworkX and/or Matplotlib not installed for visualization", "WARNING")
        return None
    
    if graph is None or graph.number_of_nodes() == 0:
        log_event("Empty graph, cannot visualize", "WARNING")
        return None
    
    output_path = output_path or str(OUTPUT_DIR / "codebase_map.png")
    
    try:
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Use spring layout for semantic distance
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
        # Size nodes by lines of code
        node_sizes = []
        for node in graph.nodes():
            loc = graph.nodes[node].get('loc', 100)
            node_sizes.append(max(100, min(3000, loc * 3)))
        
        # Color by in-degree (how many things depend on it)
        node_colors = [graph.in_degree(node) for node in graph.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10
        )
        
        # Labels for larger nodes only
        labels = {}
        for node in graph.nodes():
            loc = graph.nodes[node].get('loc', 0)
            if loc > 200:  # Only label significant modules
                short_name = node.split('.')[-1]
                labels[node] = short_name
        
        nx.draw_networkx_labels(
            graph, pos, labels, ax=ax,
            font_size=8,
            font_weight='bold'
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        fig.text(0.02, 0.02, 
                f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}\n"
                "Size = Lines of Code | Color = Dependencies (warmer = more)",
                fontsize=10, verticalalignment='bottom')
        
        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log_event(f"Saved codebase visualization to {output_path}", "INFO")
        return output_path
        
    except Exception as e:
        log_event(f"Failed to generate visualization: {e}", "ERROR")
        return None


def get_codebase_map() -> Dict[str, any]:
    """
    Returns a complete codebase map as a dictionary.
    
    This is the API for agents to introspect codebase structure.
    
    Returns:
        Dict with graph statistics and analysis
    """
    graph = build_dependency_graph()
    
    if graph is None:
        return {"error": "Could not build dependency graph"}
    
    return {
        "total_modules": graph.number_of_nodes(),
        "total_dependencies": graph.number_of_edges(),
        "circular_dependencies": find_circular_dependencies(graph),
        "clusters": get_module_clusters(graph),
        "hub_modules": get_most_connected_modules(graph, top_n=10),
        "nodes": list(graph.nodes()),
    }


# Tool function for agent use
async def visualize_codebase() -> Dict[str, any]:
    """
    Tool function: Generate and return codebase visualization.
    
    Returns:
        Dict with visualization path and codebase statistics
    """
    graph = build_dependency_graph()
    
    if graph is None:
        return {"success": False, "error": "NetworkX not available"}
    
    viz_path = generate_visualization(graph)
    stats = get_codebase_map()
    
    return {
        "success": True,
        "visualization_path": viz_path,
        "statistics": stats,
    }
