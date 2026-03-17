"""
Cache module for RAG system components.

This module provides caching functionality for various RAG components
including BM25 retrievers and other expensive operations.
"""

from .cache import CacheManager

__all__ = ['CacheManager']
