"""
Simple cache manager for RAG system components.

This provides a basic caching mechanism for expensive operations
like BM25 index loading and other retriever components.
"""

import os
import pickle
import hashlib
from typing import Any, Optional, Dict, Union
from pathlib import Path


class CacheManager:
    """Simple file-based cache manager for RAG components."""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the full path for a cache file."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached object if found, None otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for key '{key}': {e}")
                return None
        
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Store an item in cache.
        
        Args:
            key: Cache key
            value: Object to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            print(f"Error saving cache for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        cache_path = self._get_cache_path(key)
        return cache_path.exists()
    
    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting cache for key '{key}': {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Union[int, str]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = 0
        for cache_file in cache_files:
            total_size += cache_file.stat().st_size
        
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0
        }
