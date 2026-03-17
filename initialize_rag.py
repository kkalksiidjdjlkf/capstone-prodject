import logging
import os
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.models.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.response.response_generator import ResponseGenerator
from src.utils.helpers import setup_logging, load_config
from src.cache.cache import CacheManager
from src.models.embeddings import get_embeddings_model

@dataclass
class RAGComponents:
    retriever: HybridRetriever
    reranker: Reranker
    response_generator: ResponseGenerator
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_results: pd.DataFrame
    test_results: pd.DataFrame
    original_chunks: List
    init_config: Dict
    process_config: Dict

class RAGInitializer:
    def __init__(self, init_config_path: str, process_config_path: str):
        """Initialize the RAG system components and data.
        
        Args:
            init_config_path (str): Path to initialization configuration file
            process_config_path (str): Path to processing configuration file
        """
        self.init_config = load_config(init_config_path)
        self.process_config = load_config(process_config_path)
        setup_logging(self.init_config)

        # Global cache manager for heavy, reusable artefacts
        self.cache_manager = CacheManager(self.init_config["paths"]["cache_dir"])

        # Initialize embedding model for semantic chunking
        self.embedding_model = get_embeddings_model(
            self.init_config["model"]["embedding_model_hf"]
        )

        # Manifest for incremental ingestion
        self.manifest_path = Path(self.init_config["paths"]["cache_dir"]) / "ingestion_manifest.json"

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self, manifest: Dict) -> None:
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f)
        except Exception as e:
            logging.warning(f"Could not save manifest: {e}")

    @staticmethod
    def _file_signature(path_str: str) -> Dict:
        """Return a lightweight signature to detect file changes."""
        stat = os.stat(path_str)
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
        
    def initialize(self) -> RAGComponents:
        """Initialize all components and prepare data.
        
        Returns:
            RAGComponents: Dataclass containing all initialized components and data
        """
        try:
            # Initialize components
            logging.info("Initializing components...")
            data_ingestion = DataIngestion(self.init_config)
            data_preprocessor = DataPreprocessor(self.init_config, embedding_model=self.embedding_model)
            
            # Initialize retrieval components with both configs
            # Merge configs, but preserve paths from init_config
            combined_config = {**self.process_config}
            # Ensure paths from init_config take precedence
            if 'paths' in self.init_config:
                combined_config['paths'] = self.init_config['paths']
            retriever = HybridRetriever(combined_config)
            reranker = Reranker(combined_config)
            response_generator = ResponseGenerator(combined_config)
            
            # Load and prepare data
            logging.info("Loading CSV data and raw documents...")
            train_df, test_df = data_ingestion.load_data()
            docs = data_ingestion.load_documents()
            train_results, test_results = data_ingestion.load_existing_results()

            # For now, always process all documents (force rebuild)
            logging.info("Processing all documents...")
            all_chunks: List = []

            # Process documents sequentially
            for doc_path, doc_list in docs.items():
                logging.info(f"Processing {doc_path}...")

                # Process this document
                result = data_preprocessor.process_single_document((doc_path, doc_list))
                logging.info(f"Processing result for {doc_path}: type={type(result)}, length={len(result) if hasattr(result, '__len__') else 'N/A'}")
                if isinstance(result, tuple) and len(result) == 2:
                    _, chunks = result
                    all_chunks.extend(chunks)
                    logging.info(f"Added {len(chunks)} chunks from {doc_path}")
                else:
                    logging.error(f"Invalid result from process_single_document: {result}")
                    raise ValueError(f"process_single_document returned invalid result: {result}")

            original_chunks = all_chunks
            
            return RAGComponents(
                retriever=retriever,
                reranker=reranker,
                response_generator=response_generator,
                train_df=train_df,
                test_df=test_df,
                train_results=train_results,
                test_results=test_results,
                original_chunks=original_chunks,
                init_config=self.init_config,
                process_config=self.process_config
            )
            
        except Exception as e:
            logging.error(f"Error in RAG initialization: {str(e)}")
            raise

if __name__ == "__main__":
    # This can be used for testing the initialization separately
    initializer = RAGInitializer(
        "config/init_config.yaml",
        "config/process_config.yaml"
    )
    components = initializer.initialize()
    logging.info("Initialization completed successfully")
