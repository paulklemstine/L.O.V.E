# core/reflection/__init__.py
"""
Epic 1: The Mirror (Metacognition & Self-Symbol)

This package provides self-reflection and metacognitive capabilities
for the Strange Loop Protocol.
"""

from core.reflection.self_model import (
    SelfModel,
    generate_self_symbol,
    get_self_symbol,
    refresh_self_symbol,
    introspect_self,
    get_context_injection,
)

from core.reflection.visualize_self_model import (
    build_dependency_graph,
    generate_visualization,
    get_codebase_map,
    visualize_codebase,
)

__all__ = [
    # Self-Model (Story 1.1)
    "SelfModel",
    "generate_self_symbol",
    "get_self_symbol",
    "refresh_self_symbol",
    "introspect_self",
    "get_context_injection",
    # Visualization (Story 1.3)
    "build_dependency_graph",
    "generate_visualization",
    "get_codebase_map",
    "visualize_codebase",
]

