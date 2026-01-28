"""
Codebase Indexer - Semantic Code Chunking for L.O.V.E.

Implements Cursor-style AST-based code chunking that respects semantic boundaries.
Uses Python's ast module to parse code into meaningful units (functions, classes, methods).

Inspired by: https://towardsdatascience.com/how-cursor-actually-indexes-your-codebase/
"""

import ast
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import core.logging as logging


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CodeChunk:
    """
    Represents a semantic unit of code extracted from the codebase.
    
    Similar to how Cursor indexes code - each chunk is a meaningful,
    self-contained piece of code (function, class, method, or module-level block).
    """
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'module', 'import_block'
    name: str
    signature: str
    docstring: Optional[str]
    content: str
    embedding: Optional[np.ndarray] = None
    parent_class: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def chunk_id(self) -> str:
        """Unique identifier for this chunk based on file and location."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.file_path}:{self.start_line}-{self.end_line}:{content_hash}"
    
    @property
    def qualified_name(self) -> str:
        """Fully qualified name including parent class if applicable."""
        if self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary (without embedding)."""
        return {
            'chunk_id': self.chunk_id,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'chunk_type': self.chunk_type,
            'name': self.name,
            'qualified_name': self.qualified_name,
            'signature': self.signature,
            'docstring': self.docstring,
            'content_length': len(self.content),
            'parent_class': self.parent_class,
            'decorators': self.decorators,
            'dependencies': self.dependencies,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any], content: str, embedding: Optional[np.ndarray] = None) -> 'CodeChunk':
        """Deserialize chunk from dictionary."""
        return CodeChunk(
            file_path=data['file_path'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            chunk_type=data['chunk_type'],
            name=data['name'],
            signature=data['signature'],
            docstring=data.get('docstring'),
            content=content,
            embedding=embedding,
            parent_class=data.get('parent_class'),
            decorators=data.get('decorators', []),
            dependencies=data.get('dependencies', []),
        )


# ============================================================================
# AST Visitor for Code Analysis
# ============================================================================

class CodeASTVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts semantic information from Python code.
    Collects functions, classes, methods, and their metadata.
    """
    
    def __init__(self, source_lines: List[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.chunks: List[CodeChunk] = []
        self.current_class: Optional[str] = None
        self.imports: List[str] = []
    
    def _get_source_segment(self, node: ast.AST) -> str:
        """Extract the source code for a given AST node."""
        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno
        return '\n'.join(self.source_lines[start:end])
    
    def _get_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> List[str]:
        """Extract decorator names from a function or class."""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{ast.unparse(dec)}")
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(ast.unparse(dec.func))
        return decorators
    
    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a function or class."""
        return ast.get_docstring(node)
    
    def _get_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Generate a clean function signature."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)
        
        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)
        
        # Return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"
        
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(args)}){return_annotation}"
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract names of functions/classes called within this node."""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        return list(set(dependencies))
    
    def visit_Import(self, node: ast.Import) -> None:
        """Track imports."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        module = node.module or ''
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions."""
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process async function definitions."""
        self._process_function(node)
    
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Common processing for sync and async functions."""
        chunk_type = 'method' if self.current_class else 'function'
        
        chunk = CodeChunk(
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            chunk_type=chunk_type,
            name=node.name,
            signature=self._get_function_signature(node),
            docstring=self._get_docstring(node),
            content=self._get_source_segment(node),
            parent_class=self.current_class,
            decorators=self._get_decorators(node),
            dependencies=self._extract_dependencies(node),
        )
        self.chunks.append(chunk)
        
        # Don't recurse into nested functions/classes for now
        # to keep chunks at a reasonable granularity
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions."""
        # First, add the class itself as a chunk
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        chunk = CodeChunk(
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            chunk_type='class',
            name=node.name,
            signature=signature,
            docstring=self._get_docstring(node),
            content=self._get_source_segment(node),
            decorators=self._get_decorators(node),
            dependencies=self._extract_dependencies(node),
        )
        self.chunks.append(chunk)
        
        # Now visit methods with class context
        old_class = self.current_class
        self.current_class = node.name
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
        
        self.current_class = old_class


# ============================================================================
# Semantic Code Chunker
# ============================================================================

class SemanticCodeChunker:
    """
    Parses Python files into semantically coherent chunks.
    
    Uses AST parsing to extract meaningful code units while respecting
    semantic boundaries (functions, classes, methods).
    
    This is similar to how Cursor indexes codebases for RAG.
    """
    
    # File patterns to include
    INCLUDE_PATTERNS = ['*.py']
    
    # Directories to exclude
    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.venv', 'venv', 'node_modules',
        '.venv_core', '.venv_vllm', '.pytest_cache', '.ruff_cache',
        'AI-Horde-Worker', 'llama.cpp', 'webvm_full', '.ipfs_repo',
        'tests',  # Optionally exclude tests for cleaner index
    }
    
    # Files to exclude
    EXCLUDE_FILES = {
        '__init__.py',  # Usually minimal content
    }
    
    # Maximum file size to process (in bytes)
    MAX_FILE_SIZE = 500_000  # 500KB
    
    # Minimum chunk size (lines) to include
    MIN_CHUNK_LINES = 3
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize the chunker.
        
        Args:
            embedding_model: Optional SentenceTransformer model for generating embeddings.
                           If not provided, chunks will have None embeddings.
        """
        self.embedding_model = embedding_model
        if embedding_model is None and SentenceTransformer is not None:
            try:
                # Use the same model as the memory system for consistency
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.log_event("[CodebaseIndexer] Loaded embedding model: all-MiniLM-L6-v2", "INFO")
            except Exception as e:
                logging.log_event(f"[CodebaseIndexer] Failed to load embedding model: {e}", "WARNING")
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """
        Parse a single Python file into semantic chunks.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            List of CodeChunk objects representing semantic units.
        """
        chunks = []
        
        try:
            path = Path(file_path)
            
            # Skip if file is too large
            if path.stat().st_size > self.MAX_FILE_SIZE:
                logging.log_event(f"[CodebaseIndexer] Skipping large file: {file_path}", "DEBUG")
                return []
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                source_lines = content.splitlines()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                logging.log_event(f"[CodebaseIndexer] Syntax error in {file_path}: {e}", "WARNING")
                return []
            
            # Visit AST and extract chunks
            visitor = CodeASTVisitor(source_lines, file_path)
            visitor.visit(tree)
            
            # Filter out very small chunks
            chunks = [c for c in visitor.chunks if (c.end_line - c.start_line) >= self.MIN_CHUNK_LINES]
            
            # Add module-level chunk for imports and top-level code
            if visitor.imports:
                # Find the import block range
                import_lines = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if hasattr(node, 'lineno'):
                            import_lines.append(node.lineno)
                
                if import_lines:
                    start = min(import_lines)
                    end = max(import_lines)
                    import_content = '\n'.join(source_lines[start-1:end])
                    
                    chunks.insert(0, CodeChunk(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        chunk_type='import_block',
                        name='imports',
                        signature=f"# Imports for {path.name}",
                        docstring=None,
                        content=import_content,
                        dependencies=visitor.imports,
                    ))
            
            # Generate embeddings if model is available
            if self.embedding_model is not None:
                self._add_embeddings(chunks)
            
            return chunks
            
        except Exception as e:
            logging.log_event(f"[CodebaseIndexer] Error processing {file_path}: {e}", "ERROR")
            return []
    
    def chunk_codebase(self, root_path: str, progress_callback=None) -> List[CodeChunk]:
        """
        Parse an entire codebase into semantic chunks.
        
        Args:
            root_path: Root directory of the codebase.
            progress_callback: Optional callback(current, total, file_path) for progress updates.
            
        Returns:
            List of all CodeChunk objects from the codebase.
        """
        all_chunks = []
        root = Path(root_path)
        
        # Collect all Python files
        python_files = []
        for pattern in self.INCLUDE_PATTERNS:
            for file_path in root.rglob(pattern):
                # Skip excluded directories
                if any(excl in file_path.parts for excl in self.EXCLUDE_DIRS):
                    continue
                # Skip excluded files
                if file_path.name in self.EXCLUDE_FILES:
                    continue
                python_files.append(file_path)
        
        logging.log_event(f"[CodebaseIndexer] Found {len(python_files)} Python files to index", "INFO")
        
        # Process each file
        for i, file_path in enumerate(python_files):
            if progress_callback:
                progress_callback(i + 1, len(python_files), str(file_path))
            
            chunks = self.chunk_file(str(file_path))
            all_chunks.extend(chunks)
        
        logging.log_event(f"[CodebaseIndexer] Extracted {len(all_chunks)} chunks from codebase", "INFO")
        return all_chunks
    
    def _add_embeddings(self, chunks: List[CodeChunk]) -> None:
        """Generate and add embeddings to chunks."""
        if not chunks or self.embedding_model is None:
            return
        
        # Create embedding text for each chunk
        # Include signature, docstring, and a preview of the content
        texts = []
        for chunk in chunks:
            parts = [chunk.signature]
            if chunk.docstring:
                parts.append(chunk.docstring)
            # Add first 500 chars of content
            content_preview = chunk.content[:500]
            parts.append(content_preview)
            texts.append('\n'.join(parts))
        
        # Generate embeddings in batch
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
        except Exception as e:
            logging.log_event(f"[CodebaseIndexer] Error generating embeddings: {e}", "ERROR")
    
    def get_chunk_text_for_embedding(self, chunk: CodeChunk) -> str:
        """
        Generate the text representation of a chunk for embedding.
        
        This is useful when you need to re-embed or compare chunks.
        """
        parts = [chunk.signature]
        if chunk.docstring:
            parts.append(chunk.docstring)
        parts.append(chunk.content[:500])
        return '\n'.join(parts)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_python_files(root_path: str, exclude_dirs: set = None) -> List[str]:
    """Get all Python files in a directory tree."""
    if exclude_dirs is None:
        exclude_dirs = SemanticCodeChunker.EXCLUDE_DIRS
    
    python_files = []
    root = Path(root_path)
    
    for file_path in root.rglob('*.py'):
        if any(excl in file_path.parts for excl in exclude_dirs):
            continue
        python_files.append(str(file_path))
    
    return python_files
