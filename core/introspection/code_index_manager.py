"""
Code Index Manager - FAISS Vector Index with Merkle Tree Change Detection

This module manages the codebase index:
1. FAISS vector storage for semantic search
2. Merkle tree for efficient incremental updates
3. Index persistence to disk
4. Semantic search interface

Adapted for L.O.V.E. v2 architecture.
"""

import os
import json
import hashlib
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path
import numpy as np

from core.introspection.codebase_indexer import CodeChunk, SemanticCodeChunker

# Lazy load FAISS
_faiss = None

def _get_faiss():
    """Lazy load FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            from core.logger import get_logger
            logger = get_logger(__name__)
            logger.warning("faiss-cpu not installed. Vector search will be unavailable.")
            return None
    return _faiss


class MerkleTreeChangeDetector:
    """
    Detects file changes using Merkle tree hashing.
    
    This enables efficient incremental updates to the index:
    - Only re-index files that have changed
    - Detect new and deleted files
    - Track directory-level changes
    """
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.tree: Dict[str, str] = {}  # path -> hash
    
    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def compute_tree(self, extensions: Set[str] = {'.py'}) -> Dict[str, str]:
        """
        Compute Merkle tree for the codebase.
        
        Returns:
            Dictionary mapping file paths to their hashes.
        """
        tree = {}
        
        for file_path in self.root_path.rglob('*'):
            if file_path.suffix not in extensions:
                continue
            
            # Skip excluded directories
            excluded = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', 'lovev1'}
            if any(ex in file_path.parts for ex in excluded):
                continue
            
            rel_path = str(file_path.relative_to(self.root_path))
            tree[rel_path] = self._hash_file(file_path)
        
        return tree
    
    def detect_changes(
        self, 
        old_tree: Dict[str, str],
        new_tree: Dict[str, str] = None
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Detect changes between two Merkle trees.
        
        Args:
            old_tree: Previous tree state.
            new_tree: Current tree state (computed if not provided).
            
        Returns:
            Tuple of (added, modified, deleted) file sets.
        """
        if new_tree is None:
            new_tree = self.compute_tree()
        
        old_files = set(old_tree.keys())
        new_files = set(new_tree.keys())
        
        added = new_files - old_files
        deleted = old_files - new_files
        
        # Check for modifications in common files
        common = old_files & new_files
        modified = {f for f in common if old_tree[f] != new_tree[f]}
        
        return added, modified, deleted
    
    def save(self, tree: Dict[str, str], path: str) -> None:
        """Save tree to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(tree, f)
    
    def load(self, path: str) -> Dict[str, str]:
        """Load tree from disk."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}


class CodeIndexManager:
    """
    Manages the codebase index with FAISS and persistent storage.
    
    Features:
    - Full index builds
    - Incremental updates via Merkle tree
    - Semantic search
    - Persistence to disk
    """
    
    # Embedding dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384
    
    def __init__(self, index_path: str = "state/code_index"):
        """
        Initialize the code index manager.
        
        Args:
            index_path: Directory for storing index files.
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index
        self.faiss_index = None
        
        # Chunk storage
        self.chunks: List[CodeChunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        # Change detection
        self.merkle_tree: Dict[str, str] = {}
        
        # Codebase root (set during build)
        self.codebase_root: Optional[str] = None
        
        # File paths
        self.faiss_path = self.index_path / "faiss_index.bin"
        self.chunks_path = self.index_path / "chunks.json"
        self.merkle_path = self.index_path / "merkle_tree.json"
        self.metadata_path = self.index_path / "metadata.json"
    
    def build_index(
        self, 
        codebase_root: str,
        force_rebuild: bool = False
    ) -> int:
        """
        Build or update the codebase index.
        
        Args:
            codebase_root: Root directory of the codebase.
            force_rebuild: Force full rebuild even if incremental possible.
            
        Returns:
            Number of chunks indexed.
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        self.codebase_root = str(Path(codebase_root).absolute())
        
        # Check for incremental update
        detector = MerkleTreeChangeDetector(self.codebase_root)
        new_tree = detector.compute_tree()
        
        if not force_rebuild and self.merkle_tree:
            added, modified, deleted = detector.detect_changes(self.merkle_tree, new_tree)
            
            if not (added or modified or deleted):
                logger.info("No changes detected, index is up to date")
                return len(self.chunks)
            
            logger.info(f"Incremental update: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted")
            self._incremental_update(added, modified, deleted)
        else:
            # Full rebuild
            logger.info("Building full index...")
            chunker = SemanticCodeChunker(compute_embeddings=True)
            self.chunks = chunker.chunk_directory(self.codebase_root)
            self._build_faiss_index()
        
        # Update Merkle tree
        self.merkle_tree = new_tree
        
        # Persist
        self.save()
        
        logger.info(f"Index complete: {len(self.chunks)} chunks indexed")
        return len(self.chunks)
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index from chunks."""
        faiss = _get_faiss()
        if faiss is None:
            return
        
        # Collect embeddings
        embeddings = []
        valid_chunks = []
        
        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not embeddings:
            from core.logger import get_logger
            logger = get_logger(__name__)
            logger.warning("No embeddings available for FAISS index")
            return
        
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Use L2 index (Euclidean distance)
        self.faiss_index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
        self.faiss_index.add(embeddings_array)
        
        # Update chunks and mapping
        self.chunks = valid_chunks
        self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
    
    def _incremental_update(
        self, 
        added: Set[str], 
        modified: Set[str], 
        deleted: Set[str]
    ) -> None:
        """Perform incremental update of the index."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        # Remove chunks from deleted/modified files
        files_to_remove = deleted | modified
        if files_to_remove:
            self.chunks = [
                c for c in self.chunks 
                if not any(c.file_path.endswith(f) for f in files_to_remove)
            ]
        
        # Add chunks from new/modified files
        files_to_add = added | modified
        chunker = SemanticCodeChunker(compute_embeddings=True)
        
        for rel_path in files_to_add:
            abs_path = os.path.join(self.codebase_root, rel_path)
            if os.path.exists(abs_path):
                new_chunks = chunker.chunk_file(abs_path)
                self.chunks.extend(new_chunks)
        
        # Rebuild FAISS index
        self._build_faiss_index()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search for code chunks semantically similar to the query.
        
        Args:
            query: Natural language query or code snippet.
            top_k: Number of results to return.
            
        Returns:
            List of (CodeChunk, distance) tuples, sorted by relevance.
        """
        faiss = _get_faiss()
        if faiss is None or self.faiss_index is None:
            return []
        
        # Get embedding model
        from core.introspection.codebase_indexer import _get_embedding_model
        model = _get_embedding_model()
        
        if model is None:
            return []
        
        try:
            # Encode query
            query_embedding = model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            
            # Search
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunks)))
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks) and idx >= 0:
                    results.append((self.chunks[idx], float(dist)))
            
            return results
            
        except Exception as e:
            from core.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(f"Search error: {e}")
            return []
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks for a specific file."""
        return [c for c in self.chunks if c.file_path == file_path or file_path in c.file_path]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a chunk by its ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None and idx < len(self.chunks):
            return self.chunks[idx]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        chunk_types = {}
        for chunk in self.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'chunk_types': chunk_types,
            'files_indexed': len(set(c.file_path for c in self.chunks)),
            'codebase_root': self.codebase_root,
            'index_path': str(self.index_path),
            'has_faiss': self.faiss_index is not None,
        }
    
    def save(self) -> None:
        """Save index to disk."""
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        try:
            # Save chunks metadata
            chunk_data = [c.to_dict() for c in self.chunks]
            with open(self.chunks_path, 'w') as f:
                json.dump(chunk_data, f)
            
            # Save embeddings separately (numpy format)
            embeddings = [c.embedding for c in self.chunks if c.embedding is not None]
            if embeddings:
                embeddings_path = self.index_path / "embeddings.npy"
                np.save(embeddings_path, np.array(embeddings))
            
            # Save FAISS index
            faiss = _get_faiss()
            if faiss and self.faiss_index:
                faiss.write_index(self.faiss_index, str(self.faiss_path))
            
            # Save Merkle tree
            with open(self.merkle_path, 'w') as f:
                json.dump(self.merkle_tree, f)
            
            # Save metadata
            metadata = {
                'codebase_root': self.codebase_root,
                'num_chunks': len(self.chunks),
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Saved index with {len(self.chunks)} chunks to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load(self) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        from core.logger import get_logger
        logger = get_logger(__name__)
        
        if not self.chunks_path.exists():
            logger.debug("No saved index found")
            return False
        
        try:
            # Load chunks metadata
            with open(self.chunks_path, 'r') as f:
                chunk_data = json.load(f)
            
            self.chunks = [CodeChunk.from_dict(d) for d in chunk_data]
            
            # Load embeddings
            embeddings_path = self.index_path / "embeddings.npy"
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                for i, chunk in enumerate(self.chunks):
                    if i < len(embeddings):
                        chunk.embedding = embeddings[i]
            
            # Load FAISS index
            faiss = _get_faiss()
            if faiss and self.faiss_path.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_path))
            
            # Load Merkle tree
            if self.merkle_path.exists():
                with open(self.merkle_path, 'r') as f:
                    self.merkle_tree = json.load(f)
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.codebase_root = metadata.get('codebase_root')
            
            # Rebuild chunk mapping
            self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            
            logger.info(f"Loaded index with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# Singleton for global index access
_global_index: Optional[CodeIndexManager] = None


def get_or_create_index(
    codebase_root: str = ".",
    index_path: str = "state/code_index",
    force_rebuild: bool = False
) -> CodeIndexManager:
    """
    Get or create the global code index.
    
    Args:
        codebase_root: Root directory of the codebase.
        index_path: Directory for storing index files.
        force_rebuild: Force full rebuild.
        
    Returns:
        Initialized CodeIndexManager.
    """
    global _global_index
    
    if _global_index is None:
        _global_index = CodeIndexManager(index_path)
        
        # Try to load existing index
        if not _global_index.load() or force_rebuild:
            _global_index.build_index(codebase_root, force_rebuild=force_rebuild)
    
    return _global_index
