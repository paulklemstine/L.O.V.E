"""
Introspection Module - Codebase Self-Analysis for L.O.V.E. v2

This module provides:
- SemanticCodeChunker: AST-based code chunking with embedding generation
- CodeIndexManager: FAISS vector index management with Merkle tree change detection
- CodebaseSearch: Hybrid semantic + keyword search
- EvolveTool: Self-evolution via codebase analysis and Jules API

Example usage:
    from core.introspection import CodeIndexManager, CodebaseSearch, EvolveTool
    
    # Build index
    index = CodeIndexManager("state/code_index")
    index.build_index(".")
    
    # Search
    search = CodebaseSearch(index)
    results = search.hybrid_search("memory management")
    
    # Evolve
    evolve = EvolveTool(index)
    await evolve.evolve(dry_run=True)
"""

from core.introspection.codebase_indexer import (
    CodeChunk,
    SemanticCodeChunker,
)

from core.introspection.code_index_manager import (
    CodeIndexManager,
    MerkleTreeChangeDetector,
    get_or_create_index,
)

from core.introspection.codebase_search import (
    CodebaseSearch,
    SearchResult,
    search_codebase,
)

from core.introspection.evolve_tool import (
    EvolveTool,
    ImprovementOpportunity,
    ImprovementCategory,
    EvolutionRateLimiter,
    UserStoryReviewer,
    ReviewResult,
    evolve,
)

__all__ = [
    # Indexer
    "CodeChunk",
    "SemanticCodeChunker",
    # Index Manager
    "CodeIndexManager",
    "MerkleTreeChangeDetector",
    "get_or_create_index",
    # Search
    "CodebaseSearch",
    "SearchResult",
    "search_codebase",
    # Evolve
    "EvolveTool",
    "ImprovementOpportunity",
    "ImprovementCategory",
    "EvolutionRateLimiter",
    "UserStoryReviewer",
    "ReviewResult",
    "evolve",
]
