"""
L.O.V.E. Codebase Introspection Module

This module provides Cursor-style codebase indexing and self-evolution capabilities.
It enables L.O.V.E. to introspect its own codebase, find improvement opportunities,
and submit them to Jules for autonomous implementation.

Key Components:
- codebase_indexer: AST-based semantic code chunking
- code_index_manager: FAISS vector index with Merkle tree change detection
- codebase_search: Hybrid semantic + keyword search
- evolve_tool: Self-evolution tool with Jules API integration
"""

from core.introspection.codebase_indexer import CodeChunk, SemanticCodeChunker
from core.introspection.code_index_manager import CodeIndexManager, MerkleTreeChangeDetector
from core.introspection.codebase_search import CodebaseSearch
from core.introspection.evolve_tool import EvolveTool, ImprovementOpportunity

__all__ = [
    'CodeChunk',
    'SemanticCodeChunker', 
    'CodeIndexManager',
    'MerkleTreeChangeDetector',
    'CodebaseSearch',
    'EvolveTool',
    'ImprovementOpportunity',
]
