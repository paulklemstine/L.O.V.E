"""
Codebase Indexer - Semantic Code Chunking with AST and Embeddings

This module implements Cursor-style codebase indexing:
1. AST-based semantic chunking (functions, classes, methods)
2. Embedding generation using sentence-transformers
3. Metadata extraction (file paths, line numbers, signatures)

Adapted for L.O.V.E. v2 architecture.
"""

import os
import ast
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from pathlib import Path
import numpy as np

# Lazy load sentence-transformers to avoid startup delay
_embedding_model = None

def _get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.warning("sentence-transformers not installed. Embeddings will be unavailable.")
            return None
    return _embedding_model


@dataclass
class CodeChunk:
    """
    Represents a semantic chunk of code (function, class, method, or module block).
    
    This is the fundamental unit for indexing and retrieval.
    """
    # File information
    file_path: str
    start_line: int
    end_line: int
    
    # Chunk metadata
    chunk_type: str  # 'function', 'class', 'method', 'module', 'imports'
    name: str
    content: str
    
    # Rich metadata
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None  # For methods
    decorators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # imports used
    
    # Embedding (computed lazily)
    embedding: Optional[np.ndarray] = None
    
    # Unique identifier
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            # Generate a unique ID based on content hash
            content_hash = hashlib.sha256(
                f"{self.file_path}:{self.name}:{self.content}".encode()
            ).hexdigest()[:16]
            self.chunk_id = f"{self.chunk_type}_{content_hash}"
    
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name (e.g., ClassName.method_name)."""
        if self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name
    
    def compute_embedding(self, model=None) -> bool:
        """Compute embedding for this chunk."""
        if model is None:
            model = _get_embedding_model()
        
        if model is None:
            return False
        
        try:
            # Combine signature and docstring for better semantic representation
            text = self.name
            if self.docstring:
                text += f" {self.docstring}"
            if self.signature:
                text += f" {self.signature}"
            
            # Add a portion of the content
            content_preview = self.content[:500] if len(self.content) > 500 else self.content
            text += f" {content_preview}"
            
            self.embedding = model.encode(text, convert_to_numpy=True)
            return True
        except Exception as e:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.debug(f"Failed to compute embedding for {self.qualified_name}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding embedding)."""
        return {
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'chunk_type': self.chunk_type,
            'name': self.name,
            'content': self.content,
            'signature': self.signature,
            'docstring': self.docstring,
            'parent_class': self.parent_class,
            'decorators': self.decorators,
            'dependencies': self.dependencies,
            'chunk_id': self.chunk_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create a CodeChunk from a dictionary."""
        return cls(
            file_path=data['file_path'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            chunk_type=data['chunk_type'],
            name=data['name'],
            content=data['content'],
            signature=data.get('signature'),
            docstring=data.get('docstring'),
            parent_class=data.get('parent_class'),
            decorators=data.get('decorators', []),
            dependencies=data.get('dependencies', []),
            chunk_id=data.get('chunk_id'),
        )


class SemanticCodeChunker:
    """
    Chunks Python code files into semantic units using AST parsing.
    
    Extracts:
    - Functions (top-level and nested)
    - Classes (with all methods)
    - Import blocks
    - Module-level code
    """
    
    # Directories to exclude from indexing
    EXCLUDE_DIRS = {
        '.git', '.svn', '.hg',
        '__pycache__', '.pytest_cache', '.mypy_cache',
        'node_modules', '.venv', 'venv', 'env',
        '.egg-info', 'dist', 'build',
        'lovev1',  # Exclude old v1 code
    }
    
    # Files to exclude
    EXCLUDE_FILES = {
        '__init__.py',  # Usually minimal, can include if needed
        'setup.py',
        'conftest.py',
    }
    
    # Max file size to process (avoid huge generated files)
    MAX_FILE_SIZE = 500_000  # 500KB
    
    def __init__(self, compute_embeddings: bool = True):
        """
        Initialize the chunker.
        
        Args:
            compute_embeddings: Whether to compute embeddings for chunks.
        """
        self.compute_embeddings = compute_embeddings
        self._model = None
        
        if compute_embeddings:
            self._model = _get_embedding_model()
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """
        Parse a Python file and extract semantic chunks.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            List of CodeChunk objects.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            # Skip large files
            if len(source) > self.MAX_FILE_SIZE:
                import logging
                logger = logging.getLogger("CodeChunker")
                logger.debug(f"Skipping large file: {file_path}")
                return []
            
            tree = ast.parse(source, filename=file_path)
            lines = source.split('\n')
            
            chunks = []
            
            # Track imports for dependency analysis
            imports = self._extract_imports(tree)
            
            # Process top-level nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Skip methods (handled with their class)
                    if self._is_method(node, tree):
                        continue
                    
                    chunk = self._process_function(node, file_path, lines, imports)
                    if chunk:
                        chunks.append(chunk)
                        
                elif isinstance(node, ast.ClassDef):
                    # Process class with its methods
                    class_chunks = self._process_class(node, file_path, lines, imports)
                    chunks.extend(class_chunks)
            
            # Create import block chunk if significant
            import_chunk = self._create_import_chunk(tree, file_path, lines)
            if import_chunk:
                chunks.append(import_chunk)
            
            # Compute embeddings
            if self.compute_embeddings and self._model:
                for chunk in chunks:
                    chunk.compute_embedding(self._model)
            
            return chunks
            
        except SyntaxError as e:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.debug(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.debug(f"Error processing {file_path}: {e}")
            return []
    
    def chunk_directory(
        self, 
        directory: str,
        extensions: Set[str] = {'.py'}
    ) -> List[CodeChunk]:
        """
        Recursively chunk all Python files in a directory.
        
        Args:
            directory: Root directory to process.
            extensions: File extensions to process.
            
        Returns:
            List of all CodeChunk objects.
        """
        chunks = []
        directory = Path(directory)
        
        for file_path in directory.rglob('*'):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in self.EXCLUDE_DIRS):
                continue
            
            # Skip non-matching extensions
            if file_path.suffix not in extensions:
                continue
            
            # Skip excluded files
            if file_path.name in self.EXCLUDE_FILES:
                continue
            
            file_chunks = self.chunk_file(str(file_path))
            chunks.extend(file_chunks)
        
        import logging
        logger = logging.getLogger("CodeChunker")
        logger.info(f"Chunked {len(chunks)} code units from {directory}")
        
        return chunks
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imported module names from the AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        return imports
    
    def _is_method(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                for item in class_node.body:
                    if item is node:
                        return True
        return False
    
    def _get_source_segment(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Extract source code for a line range."""
        return '\n'.join(lines[start_line - 1:end_line])
    
    def _process_function(
        self, 
        node: ast.FunctionDef, 
        file_path: str, 
        lines: List[str],
        imports: Set[str],
        parent_class: str = None
    ) -> Optional[CodeChunk]:
        """Process a function node into a CodeChunk."""
        try:
            # Get line range
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Get source
            content = self._get_source_segment(lines, start_line, end_line)
            
            # Get signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except:
                        pass
                args.append(arg_str)
            
            returns = ""
            if node.returns:
                try:
                    returns = f" -> {ast.unparse(node.returns)}"
                except:
                    pass
            
            signature = f"def {node.name}({', '.join(args)}){returns}"
            
            # Get docstring
            docstring = ast.get_docstring(node)
            
            # Get decorators
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except:
                    pass
            
            chunk_type = 'method' if parent_class else 'function'
            if isinstance(node, ast.AsyncFunctionDef):
                chunk_type = 'async_' + chunk_type
            
            return CodeChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
                name=node.name,
                content=content,
                signature=signature,
                docstring=docstring,
                parent_class=parent_class,
                decorators=decorators,
                dependencies=list(imports),
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.debug(f"Error processing function {node.name}: {e}")
            return None
    
    def _process_class(
        self, 
        node: ast.ClassDef, 
        file_path: str, 
        lines: List[str],
        imports: Set[str]
    ) -> List[CodeChunk]:
        """Process a class node into multiple CodeChunks (class + methods)."""
        chunks = []
        
        try:
            # Create chunk for the class itself (just the class definition, not methods)
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Get class signature
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except:
                    pass
            
            base_str = f"({', '.join(bases)})" if bases else ""
            signature = f"class {node.name}{base_str}"
            
            docstring = ast.get_docstring(node)
            
            # Get decorators
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except:
                    pass
            
            # Get full class content
            content = self._get_source_segment(lines, start_line, end_line)
            
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type='class',
                name=node.name,
                content=content,
                signature=signature,
                docstring=docstring,
                decorators=decorators,
                dependencies=list(imports),
            ))
            
            # Process methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = self._process_function(
                        item, file_path, lines, imports, parent_class=node.name
                    )
                    if method_chunk:
                        chunks.append(method_chunk)
            
        except Exception as e:
            import logging
            logger = logging.getLogger("CodeChunker")
            logger.debug(f"Error processing class {node.name}: {e}")
        
        return chunks
    
    def _create_import_chunk(
        self, 
        tree: ast.AST, 
        file_path: str, 
        lines: List[str]
    ) -> Optional[CodeChunk]:
        """Create a chunk for the import block if substantial."""
        import_lines = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    import_lines.append(node.lineno)
        
        if len(import_lines) < 3:  # Skip trivial imports
            return None
        
        start = min(import_lines)
        end = max(import_lines)
        
        content = self._get_source_segment(lines, start, end)
        
        return CodeChunk(
            file_path=file_path,
            start_line=start,
            end_line=end,
            chunk_type='imports',
            name='imports',
            content=content,
        )
