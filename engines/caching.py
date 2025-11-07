"""Memory-efficient document and embedding cache management."""

import weakref
from typing import Any, Dict, List, Optional, Set
from threading import Lock

class DocumentCache:
    """Thread-safe weak reference cache for document chunks."""
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
    def add(self, chunk_id: str, chunk: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a document chunk to the cache with optional metadata."""
        with self._lock:
            self._cache[chunk_id] = weakref.ref(chunk, lambda _: self._cleanup(chunk_id))
            if metadata:
                self._metadata[chunk_id] = metadata
                
    def get(self, chunk_id: str) -> Optional[Any]:
        """Retrieve a document chunk from the cache."""
        with self._lock:
            ref = self._cache.get(chunk_id)
            if ref is None:
                return None
            chunk = ref()
            if chunk is None:
                self._cleanup(chunk_id)
            return chunk
            
    def _cleanup(self, chunk_id: str) -> None:
        """Clean up metadata when a chunk is garbage collected."""
        with self._lock:
            self._cache.pop(chunk_id, None)
            self._metadata.pop(chunk_id, None)

class EmbeddingCache:
    """Thread-safe LRU cache for embeddings with size limit."""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._lock = Lock()
        
    def add(self, text: str, embedding: List[float]) -> None:
        """Add an embedding to the cache with LRU eviction."""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict least recently used
                lru_key = self._access_order.pop(0)
                self._cache.pop(lru_key, None)
            
            key = self._make_key(text)
            self._cache[key] = embedding
            self._access_order.append(key)
            
    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache, updating access order."""
        key = self._make_key(text)
        with self._lock:
            embedding = self._cache.get(key)
            if embedding is not None:
                # Move to most recently used
                self._access_order.remove(key)
                self._access_order.append(key)
            return embedding
            
    @staticmethod
    def _make_key(text: str) -> str:
        """Create a cache key from text (can be enhanced with hashing if needed)."""
        return text[:100]  # Simple key strategy