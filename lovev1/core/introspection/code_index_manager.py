"""
Code Index Manager - FAISS Vector Index with Merkle Tree Change Detection

Manages the codebase vector index lifecycle:
- Builds and maintains FAISS index for semantic code search
- Uses Merkle tree for efficient change detection (like Cursor)
- Handles incremental updates when files change
- Persists index to disk for fast startup
"""

import os
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import numpy as np

# FAISS import
try:
    import faiss
except ImportError:
    faiss = None

# Sentence transformers for query embedding
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from core.introspection.codebase_indexer import (
    CodeChunk, SemanticCodeChunker, compute_file_hash, get_python_files
)
import core.logging as logging


# ============================================================================
# Merkle Tree Change Detection
# ============================================================================

class MerkleTreeChangeDetector:
    """
    Tracks file changes using Merkle tree hashes.
    
    Like Cursor, this allows efficient detection of which files have changed
    since the last indexing run. Only changed files need to be re-processed.
    """
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self._tree: Dict[str, str] = {}  # file_path -> hash
        self._root_hash: Optional[str] = None
    
    def compute_tree(self, exclude_dirs: Set[str] = None) -> Dict[str, str]:
        """
        Compute Merkle tree hashes for all Python files.
        
        Args:
            exclude_dirs: Set of directory names to exclude.
            
        Returns:
            Dict mapping file paths to their content hashes.
        """
        if exclude_dirs is None:
            exclude_dirs = SemanticCodeChunker.EXCLUDE_DIRS
        
        tree = {}
        python_files = get_python_files(self.root_path, exclude_dirs)
        
        for file_path in python_files:
            try:
                file_hash = compute_file_hash(file_path)
                # Use relative path for portability
                rel_path = os.path.relpath(file_path, self.root_path)
                tree[rel_path] = file_hash
            except Exception as e:
                logging.log_event(f"[MerkleTree] Error hashing {file_path}: {e}", "WARNING")
        
        self._tree = tree
        self._compute_root_hash()
        return tree
    
    def _compute_root_hash(self) -> None:
        """Compute the root hash of the Merkle tree."""
        if not self._tree:
            self._root_hash = None
            return
        
        # Sort keys for deterministic ordering
        sorted_items = sorted(self._tree.items())
        combined = ''.join(f"{k}:{v}" for k, v in sorted_items)
        self._root_hash = hashlib.md5(combined.encode()).hexdigest()
    
    def get_changed_files(self, old_tree: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Compare current tree to an old tree and identify changes.
        
        Args:
            old_tree: Previous Merkle tree (file_path -> hash).
            
        Returns:
            Tuple of (added_files, modified_files, deleted_files).
        """
        current_files = set(self._tree.keys())
        old_files = set(old_tree.keys())
        
        added = list(current_files - old_files)
        deleted = list(old_files - current_files)
        
        # Check for modifications among files that exist in both
        modified = []
        for file_path in current_files & old_files:
            if self._tree[file_path] != old_tree[file_path]:
                modified.append(file_path)
        
        return added, modified, deleted
    
    def has_changes(self, old_root_hash: str) -> bool:
        """Quick check if anything has changed."""
        return self._root_hash != old_root_hash
    
    @property
    def root_hash(self) -> Optional[str]:
        return self._root_hash
    
    @property
    def tree(self) -> Dict[str, str]:
        return self._tree.copy()


# ============================================================================
# Code Index Manager
# ============================================================================

class CodeIndexManager:
    """
    Manages the codebase FAISS vector index.
    
    Features:
    - Semantic code search using FAISS
    - Merkle tree-based change detection
    - Incremental index updates
    - Persistent storage
    """
    
    # Index configuration
    INDEX_VERSION = "1.0"
    EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
    
    def __init__(self, index_path: str = "state/code_index"):
        """
        Initialize the code index manager.
        
        Args:
            index_path: Directory path for storing index files.
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.chunker: Optional[SemanticCodeChunker] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        
        # Chunk storage
        self.chunks: List[CodeChunk] = []
        self.chunk_id_to_index: Dict[str, int] = {}  # chunk_id -> faiss index
        
        # Merkle tree for change detection
        self.merkle_tree: Dict[str, str] = {}
        self.root_hash: Optional[str] = None
        
        # Metadata
        self.last_indexed: Optional[float] = None
        self.codebase_root: Optional[str] = None
        
        # Initialize embedding model
        self._init_embedding_model()
    
    def _init_embedding_model(self) -> None:
        """Initialize the embedding model."""
        if SentenceTransformer is None:
            logging.log_event("[CodeIndexManager] sentence-transformers not available", "WARNING")
            return
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.chunker = SemanticCodeChunker(self.embedding_model)
            logging.log_event("[CodeIndexManager] Initialized with all-MiniLM-L6-v2", "INFO")
        except Exception as e:
            logging.log_event(f"[CodeIndexManager] Failed to load model: {e}", "ERROR")
    
    def build_index(self, codebase_path: str, progress_callback=None) -> int:
        """
        Build a fresh index of the entire codebase.
        
        Args:
            codebase_path: Root path of the codebase to index.
            progress_callback: Optional callback(current, total, file_path).
            
        Returns:
            Number of chunks indexed.
        """
        if self.chunker is None:
            logging.log_event("[CodeIndexManager] Cannot build index: chunker not initialized", "ERROR")
            return 0
        
        self.codebase_root = str(Path(codebase_path).resolve())
        logging.log_event(f"[CodeIndexManager] Building index for {self.codebase_root}", "INFO")
        
        start_time = time.time()
        
        # Chunk the codebase
        self.chunks = self.chunker.chunk_codebase(codebase_path, progress_callback)
        
        if not self.chunks:
            logging.log_event("[CodeIndexManager] No chunks extracted from codebase", "WARNING")
            return 0
        
        # Build FAISS index
        self._build_faiss_index()
        
        # Compute Merkle tree
        detector = MerkleTreeChangeDetector(self.codebase_root)
        self.merkle_tree = detector.compute_tree()
        self.root_hash = detector.root_hash
        
        self.last_indexed = time.time()
        elapsed = time.time() - start_time
        
        logging.log_event(
            f"[CodeIndexManager] Indexed {len(self.chunks)} chunks in {elapsed:.2f}s",
            "INFO"
        )
        
        # Auto-save
        self.save()
        
        return len(self.chunks)
    
    def _build_faiss_index(self) -> None:
        """Build the FAISS index from chunks."""
        if faiss is None:
            logging.log_event("[CodeIndexManager] FAISS not available", "ERROR")
            return
        
        # Collect embeddings
        embeddings = []
        self.chunk_id_to_index = {}
        
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                self.chunk_id_to_index[chunk.chunk_id] = len(embeddings) - 1
        
        if not embeddings:
            logging.log_event("[CodeIndexManager] No embeddings to index", "WARNING")
            return
        
        # Build index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Use L2 (Euclidean) distance
        self.faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.faiss_index.add(embeddings_array)
        
        logging.log_event(
            f"[CodeIndexManager] Built FAISS index with {self.faiss_index.ntotal} vectors",
            "INFO"
        )
    
    def update_index(self, progress_callback=None) -> Tuple[int, int, int]:
        """
        Incrementally update the index for changed files.
        
        Returns:
            Tuple of (added, modified, deleted) file counts.
        """
        if not self.codebase_root:
            logging.log_event("[CodeIndexManager] No codebase root set, cannot update", "WARNING")
            return (0, 0, 0)
        
        # Compute current Merkle tree
        detector = MerkleTreeChangeDetector(self.codebase_root)
        current_tree = detector.compute_tree()
        
        # Check for changes
        if not detector.has_changes(self.root_hash):
            logging.log_event("[CodeIndexManager] No changes detected", "INFO")
            return (0, 0, 0)
        
        # Get changed files
        added, modified, deleted = detector.get_changed_files(self.merkle_tree)
        
        logging.log_event(
            f"[CodeIndexManager] Changes: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted",
            "INFO"
        )
        
        # Remove chunks from deleted and modified files
        files_to_remove = set(deleted + modified)
        self.chunks = [c for c in self.chunks if 
                      os.path.relpath(c.file_path, self.codebase_root) not in files_to_remove]
        
        # Add chunks from added and modified files
        files_to_add = added + modified
        for i, rel_path in enumerate(files_to_add):
            if progress_callback:
                progress_callback(i + 1, len(files_to_add), rel_path)
            
            abs_path = os.path.join(self.codebase_root, rel_path)
            new_chunks = self.chunker.chunk_file(abs_path)
            self.chunks.extend(new_chunks)
        
        # Rebuild FAISS index
        self._build_faiss_index()
        
        # Update Merkle tree
        self.merkle_tree = current_tree
        self.root_hash = detector.root_hash
        self.last_indexed = time.time()
        
        # Auto-save
        self.save()
        
        return (len(added), len(modified), len(deleted))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search for code chunks semantically similar to the query.
        
        Args:
            query: Natural language query or code snippet.
            top_k: Number of results to return.
            
        Returns:
            List of (CodeChunk, distance) tuples, sorted by relevance.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logging.log_event("[CodeIndexManager] Index is empty", "WARNING")
            return []
        
        if self.embedding_model is None:
            logging.log_event("[CodeIndexManager] No embedding model available", "ERROR")
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Map indices back to chunks
        results = []
        index_to_chunk = {v: k for k, v in self.chunk_id_to_index.items()}
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue
            
            # Find chunk by index
            for chunk in self.chunks:
                if self.chunk_id_to_index.get(chunk.chunk_id) == idx:
                    results.append((chunk, float(dist)))
                    break
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Retrieve a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks from a specific file."""
        return [c for c in self.chunks if c.file_path == file_path]
    
    def save(self) -> None:
        """Persist the index to disk."""
        if not self.chunks:
            logging.log_event("[CodeIndexManager] Nothing to save", "DEBUG")
            return
        
        try:
            # Save FAISS index
            if self.faiss_index is not None and faiss is not None:
                faiss_path = self.index_path / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_path))
            
            # Save chunk metadata (without embeddings and content)
            metadata = {
                'version': self.INDEX_VERSION,
                'codebase_root': self.codebase_root,
                'last_indexed': self.last_indexed,
                'chunk_count': len(self.chunks),
                'chunks': [c.to_dict() for c in self.chunks],
                'chunk_id_to_index': self.chunk_id_to_index,
            }
            
            with open(self.index_path / "chunk_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save chunk contents separately (for retrieval)
            contents = {c.chunk_id: c.content for c in self.chunks}
            with open(self.index_path / "chunk_contents.json", 'w') as f:
                json.dump(contents, f)
            
            # Save Merkle tree
            merkle_data = {
                'root_hash': self.root_hash,
                'tree': self.merkle_tree,
            }
            with open(self.index_path / "merkle_tree.json", 'w') as f:
                json.dump(merkle_data, f)
            
            logging.log_event(f"[CodeIndexManager] Saved index to {self.index_path}", "INFO")
            
        except Exception as e:
            logging.log_event(f"[CodeIndexManager] Failed to save index: {e}", "ERROR")
    
    def load(self) -> bool:
        """
        Load a previously saved index from disk.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            metadata_path = self.index_path / "chunk_metadata.json"
            if not metadata_path.exists():
                logging.log_event("[CodeIndexManager] No saved index found", "INFO")
                return False
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check version compatibility
            if metadata.get('version') != self.INDEX_VERSION:
                logging.log_event("[CodeIndexManager] Index version mismatch, rebuild required", "WARNING")
                return False
            
            self.codebase_root = metadata['codebase_root']
            self.last_indexed = metadata['last_indexed']
            self.chunk_id_to_index = metadata['chunk_id_to_index']
            
            # Load chunk contents
            with open(self.index_path / "chunk_contents.json", 'r') as f:
                contents = json.load(f)
            
            # Load FAISS index
            faiss_path = self.index_path / "faiss_index.bin"
            if faiss_path.exists() and faiss is not None:
                self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Reconstruct chunks
            self.chunks = []
            for chunk_data in metadata['chunks']:
                chunk_id = chunk_data.get('chunk_id', 
                    f"{chunk_data['file_path']}:{chunk_data['start_line']}-{chunk_data['end_line']}")
                content = contents.get(chunk_id, "")
                
                # Load embedding from FAISS if available
                embedding = None
                if self.faiss_index is not None and chunk_id in self.chunk_id_to_index:
                    idx = self.chunk_id_to_index[chunk_id]
                    if idx < self.faiss_index.ntotal:
                        embedding = self.faiss_index.reconstruct(idx)
                
                chunk = CodeChunk.from_dict(chunk_data, content, embedding)
                self.chunks.append(chunk)
            
            # Load Merkle tree
            merkle_path = self.index_path / "merkle_tree.json"
            if merkle_path.exists():
                with open(merkle_path, 'r') as f:
                    merkle_data = json.load(f)
                self.root_hash = merkle_data.get('root_hash')
                self.merkle_tree = merkle_data.get('tree', {})
            
            logging.log_event(
                f"[CodeIndexManager] Loaded index with {len(self.chunks)} chunks",
                "INFO"
            )
            return True
            
        except Exception as e:
            logging.log_event(f"[CodeIndexManager] Failed to load index: {e}", "ERROR")
            return False
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        file_stats = {}
        for chunk in self.chunks:
            file_path = chunk.file_path
            if file_path not in file_stats:
                file_stats[file_path] = {'chunks': 0, 'types': set()}
            file_stats[file_path]['chunks'] += 1
            file_stats[file_path]['types'].add(chunk.chunk_type)
        
        type_counts = {}
        for chunk in self.chunks:
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'total_files': len(file_stats),
            'chunk_types': type_counts,
            'index_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'codebase_root': self.codebase_root,
            'last_indexed': self.last_indexed,
            'root_hash': self.root_hash,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def get_or_create_index(codebase_path: str, index_path: str = "state/code_index") -> CodeIndexManager:
    """
    Get an existing index or create a new one.
    
    Args:
        codebase_path: Root path of the codebase.
        index_path: Path for storing the index.
        
    Returns:
        Initialized CodeIndexManager.
    """
    manager = CodeIndexManager(index_path)
    
    if manager.load():
        # Check if update is needed
        manager.update_index()
    else:
        # Build fresh index
        manager.build_index(codebase_path)
    
    return manager
