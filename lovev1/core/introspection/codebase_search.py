"""
Codebase Search - Hybrid Semantic + Keyword Search

Provides a unified search interface combining:
- Semantic search using FAISS vector embeddings
- Keyword search using ripgrep/grep patterns
- Context assembly for LLM queries
"""

import os
import re
import subprocess
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from core.introspection.codebase_indexer import CodeChunk
from core.introspection.code_index_manager import CodeIndexManager
import core.logging as logging


@dataclass
class SearchResult:
    """Unified search result from any search method."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float  # Lower is better for semantic, higher for keyword
    source: str  # 'semantic', 'keyword', 'hybrid'
    chunk_type: Optional[str] = None
    name: Optional[str] = None
    signature: Optional[str] = None
    
    def to_context_string(self, max_lines: int = 50) -> str:
        """Format result for LLM context."""
        lines = self.content.split('\n')
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append("... (truncated)")
        
        content = '\n'.join(lines)
        header = f"# {self.file_path}:{self.start_line}-{self.end_line}"
        if self.name:
            header += f" ({self.chunk_type}: {self.name})"
        
        return f"{header}\n```python\n{content}\n```"


class CodebaseSearch:
    """
    Hybrid semantic + keyword search for the codebase.
    
    Combines vector-based semantic search with traditional keyword matching
    for comprehensive code retrieval.
    """
    
    def __init__(self, code_index: CodeIndexManager):
        """
        Initialize the search engine.
        
        Args:
            code_index: Initialized CodeIndexManager with indexed codebase.
        """
        self.code_index = code_index
        self.codebase_root = code_index.codebase_root
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using semantic similarity.
        
        Args:
            query: Natural language query or code snippet.
            top_k: Maximum number of results.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        results = []
        
        # Use the code index for semantic search
        matches = self.code_index.search(query, top_k)
        
        for chunk, distance in matches:
            results.append(SearchResult(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                score=distance,
                source='semantic',
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                signature=chunk.signature,
            ))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using keyword/pattern matching.
        
        Uses ripgrep if available, falls back to grep or Python search.
        
        Args:
            query: Search pattern (supports regex).
            top_k: Maximum number of results.
            
        Returns:
            List of SearchResult objects.
        """
        if not self.codebase_root:
            return []
        
        results = []
        
        # Try ripgrep first (fastest)
        rg_results = self._ripgrep_search(query, top_k * 2)
        if rg_results:
            results = rg_results[:top_k]
        else:
            # Fallback to Python-based search
            results = self._python_search(query, top_k)
        
        return results
    
    def _ripgrep_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using ripgrep."""
        try:
            cmd = [
                'rg',
                '--json',
                '--max-count', str(max_results),
                '--type', 'py',
                '--ignore-case',
                query,
                self.codebase_root
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode not in [0, 1]:  # 1 means no matches
                return []
            
            results = []
            import json
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('type') == 'match':
                        match_data = data['data']
                        file_path = match_data['path']['text']
                        line_num = match_data['line_number']
                        content = match_data['lines']['text'].strip()
                        
                        results.append(SearchResult(
                            file_path=file_path,
                            start_line=line_num,
                            end_line=line_num,
                            content=content,
                            score=1.0,  # Keyword matches are equally weighted
                            source='keyword',
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return results
            
        except FileNotFoundError:
            logging.log_event("[CodebaseSearch] ripgrep not found, using fallback", "DEBUG")
            return []
        except Exception as e:
            logging.log_event(f"[CodebaseSearch] ripgrep error: {e}", "WARNING")
            return []
    
    def _python_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Fallback Python-based search."""
        results = []
        pattern = re.compile(query, re.IGNORECASE)
        
        for chunk in self.code_index.chunks:
            matches = list(pattern.finditer(chunk.content))
            if matches:
                # Score based on number of matches
                score = 1.0 / (1 + len(matches))
                
                results.append(SearchResult(
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=score,
                    source='keyword',
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                ))
        
        # Sort by score (lower is better)
        results.sort(key=lambda r: r.score)
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Combine semantic and keyword search for best results.
        
        Uses reciprocal rank fusion to combine results from both methods.
        
        Args:
            query: Search query (can be natural language or code).
            top_k: Maximum number of results.
            
        Returns:
            List of SearchResult objects with combined ranking.
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)
        
        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        k = 60  # RRF constant
        
        # Process semantic results
        for rank, result in enumerate(semantic_results):
            key = f"{result.file_path}:{result.start_line}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            result_map[key] = result
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            key = f"{result.file_path}:{result.start_line}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            
            # Prefer semantic result if we have both (has more metadata)
            if key not in result_map:
                result_map[key] = result
        
        # Sort by combined RRF score (higher is better)
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        combined_results = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.source = 'hybrid'
            result.score = rrf_scores[key]
            combined_results.append(result)
        
        return combined_results
    
    def find_function(self, function_name: str) -> Optional[SearchResult]:
        """Find a specific function by name."""
        for chunk in self.code_index.chunks:
            if chunk.chunk_type in ['function', 'method'] and chunk.name == function_name:
                return SearchResult(
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=0.0,
                    source='exact',
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    signature=chunk.signature,
                )
        return None
    
    def find_class(self, class_name: str) -> Optional[SearchResult]:
        """Find a specific class by name."""
        for chunk in self.code_index.chunks:
            if chunk.chunk_type == 'class' and chunk.name == class_name:
                return SearchResult(
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=0.0,
                    source='exact',
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    signature=chunk.signature,
                )
        return None
    
    def get_file_contents(self, file_path: str) -> str:
        """Get the full contents of a file."""
        chunks = self.code_index.get_chunks_by_file(file_path)
        if not chunks:
            # Try reading directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                return ""
        
        # Sort by start line and combine
        chunks.sort(key=lambda c: c.start_line)
        return '\n\n'.join(c.content for c in chunks)
    
    def assemble_context(
        self,
        results: List[SearchResult],
        max_tokens: int = 8000,
        chars_per_token: float = 4.0
    ) -> str:
        """
        Assemble search results into a context string for LLM.
        
        Args:
            results: List of search results to include.
            max_tokens: Maximum approximate tokens to include.
            chars_per_token: Approximate characters per token.
            
        Returns:
            Formatted context string.
        """
        max_chars = int(max_tokens * chars_per_token)
        context_parts = []
        current_chars = 0
        
        for result in results:
            part = result.to_context_string()
            if current_chars + len(part) > max_chars:
                break
            context_parts.append(part)
            current_chars += len(part)
        
        return '\n\n---\n\n'.join(context_parts)
    
    def search_with_context(
        self,
        query: str,
        search_type: str = 'hybrid',
        top_k: int = 5,
        max_context_tokens: int = 4000
    ) -> Tuple[List[SearchResult], str]:
        """
        Search and return both results and assembled context.
        
        Args:
            query: Search query.
            search_type: 'semantic', 'keyword', or 'hybrid'.
            top_k: Number of results.
            max_context_tokens: Max tokens for context.
            
        Returns:
            Tuple of (results, context_string).
        """
        if search_type == 'semantic':
            results = self.semantic_search(query, top_k)
        elif search_type == 'keyword':
            results = self.keyword_search(query, top_k)
        else:
            results = self.hybrid_search(query, top_k)
        
        context = self.assemble_context(results, max_context_tokens)
        return results, context


# ============================================================================
# Tool Interface Functions
# ============================================================================

async def search_codebase(
    query: str,
    search_type: str = 'hybrid',
    top_k: int = 5,
    codebase_path: str = None,
    index_path: str = 'state/code_index'
) -> str:
    """
    Search the codebase using semantic similarity.
    
    This is the tool interface for the search functionality.
    
    Args:
        query: Natural language description of what to find.
        search_type: 'semantic', 'keyword', or 'hybrid'.
        top_k: Number of results to return.
        codebase_path: Root path of codebase (uses index's stored path if None).
        index_path: Path to the code index.
        
    Returns:
        Formatted string with relevant code snippets.
    """
    from core.introspection.code_index_manager import get_or_create_index
    
    # Get or create index
    if codebase_path is None:
        codebase_path = os.getcwd()
    
    index = get_or_create_index(codebase_path, index_path)
    search = CodebaseSearch(index)
    
    # Perform search
    results, context = search.search_with_context(
        query=query,
        search_type=search_type,
        top_k=top_k,
    )
    
    if not results:
        return f"No results found for query: {query}"
    
    # Format output
    output = [f"Found {len(results)} relevant code snippets for: '{query}'\n"]
    output.append(context)
    
    return '\n'.join(output)
