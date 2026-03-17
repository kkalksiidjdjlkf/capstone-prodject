"""
Embeddings model management with caching support.

This module provides centralized management of embedding models with
LRU caching to improve performance by avoiding repeated model loading.
"""

from functools import lru_cache
from typing import Any
from langchain_ollama import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings_model(model_name: str = "bge-m3:latest") -> OllamaEmbeddings:
    """
    Get cached embeddings model instance.

    Uses LRU caching to ensure only one model instance exists in memory
    at any time, dramatically improving performance for repeated queries.

    Args:
        model_name (str): Name of the Ollama model to use

    Returns:
        OllamaEmbeddings: Cached model instance

    Example:
        >>> model = get_embeddings_model()
        >>> embeddings = model.embed_documents(["Hello world"])
    """
    logger.info(f"Loading embeddings model: {model_name}")

    model = OllamaEmbeddings(
        model=model_name,
        base_url="http://host.docker.internal:11434"
    )

    logger.info(f"Embeddings model loaded successfully: {model_name}")
    return model


def clear_embeddings_cache():
    """
    Clear the embeddings model cache.
    
    Useful for testing or when you need to force model reload.
    """
    get_embeddings_model.cache_clear()
    logger.info("Embeddings model cache cleared")
