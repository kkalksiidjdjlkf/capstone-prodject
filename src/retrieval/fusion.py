"""
RAG Fusion implementation for query translation and result fusion.
Combines multiple retrieval strategies to improve cross-language and multilingual retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGFusion:
    """
    Implements Reciprocal Rank Fusion for combining results from multiple retrieval strategies.
    Supports query translation and multilingual retrieval fusion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG Fusion with configuration.

        Args:
            config: Configuration dictionary containing fusion parameters
        """
        self.config = config
        self.rrf_k = config.get('fusion', {}).get('rrf_k', 60)
        self.logger = logging.getLogger(__name__)

    def reciprocal_rank_fusion(self, results_list: List[List[Tuple[Document, float]]],
                              k: int = None) -> List[Tuple[Document, float]]:
        """
        Apply Reciprocal Rank Fusion to combine multiple ranked result lists.

        Args:
            results_list: List of result lists, each containing (document, score) tuples
            k: RRF constant (default: 60, empirical sweet spot)

        Returns:
            Fused and reranked list of (document, score) tuples
        """
        if k is None:
            k = self.rrf_k

        # Create unique document identifier -> scores mapping
        doc_scores = defaultdict(float)

        for results in results_list:
            for rank, (doc, _) in enumerate(results):
                # Create unique identifier for document
                doc_id = self._get_document_id(doc)

                # Apply RRF formula: score += 1 / (k + rank)
                doc_scores[doc_id] += 1.0 / (k + rank)

        # Sort by fused scores (highest first)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Convert back to (document, score) tuples, preserving original documents
        fused_results = []
        doc_map = self._create_document_map(results_list)

        for doc_id, fused_score in sorted_docs:
            if doc_id in doc_map:
                # Use the document from the first result list that contains it
                original_doc = doc_map[doc_id]
                fused_results.append((original_doc, fused_score))

        self.logger.info(f"Fused {len(results_list)} result lists into {len(fused_results)} unique documents")
        return fused_results

    def _get_document_id(self, doc: Document) -> str:
        """
        Create a unique identifier for a document based on its content and metadata.

        Args:
            doc: Document to identify

        Returns:
            Unique string identifier
        """
        # Use chunk content hash + source for uniqueness
        content_hash = hash(doc.page_content.strip())
        source = doc.metadata.get('source', 'unknown')
        chunk_id = doc.metadata.get('chunk_id', str(content_hash))

        return f"{source}:{chunk_id}"

    def _create_document_map(self, results_list: List[List[Tuple[Document, float]]]) -> Dict[str, Document]:
        """
        Create mapping from document IDs to original documents.

        Args:
            results_list: List of result lists

        Returns:
            Dictionary mapping document IDs to Document objects
        """
        doc_map = {}

        for results in results_list:
            for doc, _ in results:
                doc_id = self._get_document_id(doc)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        return doc_map

    def fuse_with_weights(self, results_list: List[List[Tuple[Document, float]]],
                         weights: List[float] = None) -> List[Tuple[Document, float]]:
        """
        Fuse results with custom weights instead of RRF.

        Args:
            results_list: List of result lists
            weights: Weights for each result list (must sum to 1.0)

        Returns:
            Weighted fused results
        """
        if weights is None:
            # Equal weights
            weights = [1.0 / len(results_list)] * len(results_list)
        elif abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        if len(weights) != len(results_list):
            raise ValueError("Number of weights must match number of result lists")

        # Normalize scores within each result list
        normalized_results = []
        for results in results_list:
            if not results:
                normalized_results.append([])
                continue

            # Get max score for normalization
            max_score = max(score for _, score in results)

            # Normalize to [0, 1] range
            normalized = []
            for doc, score in results:
                norm_score = score / max_score if max_score > 0 else 0.0
                normalized.append((doc, norm_score))
            normalized_results.append(normalized)

        # Apply weights and combine
        doc_scores = defaultdict(float)

        for i, results in enumerate(normalized_results):
            weight = weights[i]
            for doc, norm_score in results:
                doc_id = self._get_document_id(doc)
                doc_scores[doc_id] += weight * norm_score

        # Sort and return
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        doc_map = self._create_document_map(results_list)

        fused_results = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_map:
                fused_results.append((doc_map[doc_id], score))

        return fused_results

    def deduplicate_results(self, results: List[Tuple[Document, float]],
                           similarity_threshold: float = 0.95) -> List[Tuple[Document, float]]:
        """
        Remove duplicate or highly similar documents from results.

        Args:
            results: List of (document, score) tuples
            similarity_threshold: Threshold for considering documents duplicates

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        # For now, use simple content-based deduplication
        # In a full implementation, you'd use embedding similarity
        seen_content = set()
        deduplicated = []

        for doc, score in results:
            content_hash = hash(doc.page_content.strip())

            # Simple deduplication based on content hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append((doc, score))

        self.logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)} unique documents")
        return deduplicated


class QueryVariantGenerator:
    """
    Generates multiple query variants for fusion retrieval.
    Supports translation, paraphrasing, and query expansion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize query variant generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Import here to avoid circular imports
        from ..response.response_generator import ResponseGenerator

        # Create a minimal response generator for query operations
        self.response_generator = ResponseGenerator(config)

    def generate_variants(self, query: str, num_variants: int = 5) -> List[str]:
        """
        Generate multiple query variants for fusion.

        Args:
            query: Original query
            num_variants: Number of variants to generate (including original)

        Returns:
            List of query variants
        """
        variants = [query]  # Always include original

        # Generate translations if enabled
        if self.config.get('fusion', {}).get('translation_enabled', True):
            translations = self._generate_translations(query)
            variants.extend(translations)

        # Generate paraphrases if enabled
        if self.config.get('fusion', {}).get('paraphrase_enabled', True):
            paraphrases = self._generate_paraphrases(query)
            variants.extend(paraphrases)

        # Remove duplicates and limit to requested number
        unique_variants = list(dict.fromkeys(variants))  # Preserve order
        return unique_variants[:num_variants]

    def _generate_translations(self, query: str) -> List[str]:
        """
        Generate translated versions of the query.

        Args:
            query: Original query

        Returns:
            List of translated queries
        """
        translations = []

        try:
            # Use existing translation method from response generator
            translated = self.response_generator.expand_query(query)
            if translated and translated != query:
                translations.append(translated)

        except Exception as e:
            self.logger.warning(f"Translation generation failed: {e}")

        return translations

    def _generate_paraphrases(self, query: str, max_paraphrases: int = 2) -> List[str]:
        """
        Generate paraphrased versions of the query.

        Args:
            query: Original query
            max_paraphrases: Maximum number of paraphrases to generate

        Returns:
            List of paraphrased queries
        """
        paraphrases = []

        try:
            # Use LLM to generate paraphrases
            paraphrase_prompt = f"""
            Generate {max_paraphrases} different paraphrases of this query.
            Each paraphrase should convey the same meaning but use different words.
            Keep them concise and natural.

            Original query: {query}

            Paraphrases (one per line):
            """

            # This would need to be implemented using the LLM
            # For now, return empty list as placeholder
            self.logger.debug("Paraphrase generation not yet implemented")

        except Exception as e:
            self.logger.warning(f"Paraphrase generation failed: {e}")

        return paraphrases
