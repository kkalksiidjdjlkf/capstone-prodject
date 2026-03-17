from typing import List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import logging
from urllib.parse import urlparse
from tqdm import tqdm
from langchain_core.documents import Document
from .loaders import FileLoader
from ..cache.cache import CacheManager

class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.cache_manager = CacheManager(self.config['paths']['cache_dir'])
        self.file_loader = FileLoader(config)
        self.logger = logging.getLogger(__name__)
    
    def get_document_paths(self) -> List[Path]:
        """Resolves directories into a list of individual files."""
        raw_paths = self.config['files']['document_paths']
        resolved_files = []
        
        for p in raw_paths:
            # Handle Windows-style paths if they exist in config
            p = p.replace('D:/Secure-Offline-RAG-System/', './') if 'D:' in p else p
            path_obj = Path(p)
            
            if path_obj.exists():
                if path_obj.is_dir():
                    self.logger.info(f"Scanning directory: {p}")
                    # Recursively find all files, ignore hidden system files (like .DS_Store)
                    for file in path_obj.rglob('*'):
                        if file.is_file() and not file.name.startswith('.'):
                            resolved_files.append(file)
                else:
                    if not path_obj.name.startswith('.'):
                        resolved_files.append(path_obj)
            else:
                self.logger.warning(f"Path not found: {p}")
                
        return resolved_files
    
    def load_documents(self, traverse_toc: bool = True) -> List[Document]:
        # Step 1: Get the actual list of files (filtering out directories)
        files_to_process = self.get_document_paths()
        all_documents = {}
        
        if not files_to_process:
            self.logger.error("No valid documents found in the specified paths.")
            return {}

        # Step 2: Process each file individually
        for file_path in tqdm(files_to_process, desc="Loading documents"):
            source_str = str(file_path)
            
            # Check Cache first
            cached_content = self.cache_manager.get(source_str)
            if cached_content is not None:
                all_documents[source_str] = cached_content
                continue
            
            try:
                # IMPORTANT: Only pass actual files to the loader
                docs = self.file_loader.load(file_path)
                if docs:
                    self.cache_manager.set(source_str, docs)
                    all_documents[source_str] = docs
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return all_documents

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Loading CSV data...")
        train_df = pd.read_csv(self.config['files']['train_data'])
        test_df = pd.read_csv(self.config['files']['test_data'])
        return train_df, test_df

    def load_existing_results(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        train_res, test_res = None, None
        try:
            t_path = Path(self.config['files'].get('train_results', ''))
            te_path = Path(self.config['files'].get('test_results', ''))
            if t_path.exists(): train_res = pd.read_csv(t_path)
            if te_path.exists(): test_res = pd.read_csv(te_path)
        except: pass
        return train_res, test_res

    def save_results(self, results: pd.DataFrame, is_test: bool = False):
        """
        Save evaluation results to the configured CSV path.

        Note:
            Uses pathlib.Path instead of the invalid pd.Path helper.
        """
        from pathlib import Path

        path_str = self.config['files']['test_results'] if is_test else self.config['files']['train_results']
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
