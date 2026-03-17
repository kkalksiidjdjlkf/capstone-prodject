from typing import Dict, Type, Callable, List, Any, Union
from pathlib import Path
import logging
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    BSHTMLLoader,
    UnstructuredEmailLoader,
    NotebookLoader,
    PyMuPDFLoader
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser

from .base import BaseLoader
from .file_converter import OfficeConverter

class FileLoader(BaseLoader):
    """
    Unified file loader for technical RAG. 
    Optimized for PyMuPDF to handle large technical specifications.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.file_converter = OfficeConverter()
        self.config = config
        
        # Optimize CPU threads for processing
        num_threads = self.config.get("processing", {}).get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = str(num_threads if num_threads else max(1, os.cpu_count() - 1))
        
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Maps file extensions to specific handler methods or classes."""
        self.document_loaders: Dict[str, Union[Type, Callable]] = {
            # Text and Markdown
            '.txt': TextLoader,
            '.md': TextLoader,
            
            # PDF Documents (Unified Handler)
            '.pdf': self._load_pdf,
            
            # Office Documents
            '.docx': self._convert_and_load,
            '.doc': self._convert_and_load,
            '.ppt': self._convert_and_load,
            '.pptx': self._convert_and_load,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.odt': self._convert_and_load,
            
            # Data Files
            '.csv': self._load_csv,
            '.json': JSONLoader,
            
            # Web and Email
            '.html': BSHTMLLoader,
            '.htm': BSHTMLLoader,
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader,
            
            # Notebooks and Code
            '.ipynb': self._load_jupyter_notebook,
            '.py': self._load_code_file,
            '.js': self._load_code_file,
            '.ts': self._load_code_file,
            '.sql': self._load_code_file,
            '.cpp': self._load_code_file,
            '.h': self._load_code_file,
        }

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Directly uses PyMuPDF for fast, page-by-page extraction."""
        try:
            self.logger.info(f"Using standard PyMuPDF loader for {file_path}")
            loader = PyMuPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            self.logger.error(f"PyMuPDF failed on {file_path}: {str(e)}")
            return []

    def _load_csv(self, file_path: str) -> List[Document]:
        """Loads CSV and handles column filtering based on config."""
        try:
            ignore_columns = self.config.get("ingestion", {}).get("ignore_columns", [])
            df = pd.read_csv(file_path)
            content_columns = [col for col in df.columns if col not in ignore_columns]
            
            loader = CSVLoader(file_path=file_path, content_columns=content_columns)
            return loader.load()
        except Exception as e:
            self.logger.error(f"CSV Load Error: {e}")
            return []

    def _load_excel(self, file_path: str) -> List[Document]:
        """Loads Excel files and converts HTML tables to Markdown."""
        try:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            docs = loader.load()
            processed_docs = []
            for doc in docs:
                if "text_as_html" in doc.metadata:
                    content = self.html_table_to_markdown(doc.metadata["text_as_html"])
                    doc.page_content = content
                    doc.metadata["is_markdown"] = True
                processed_docs.append(doc)
            return processed_docs
        except Exception as e:
            self.logger.error(f"Excel Error: {e}")
            return []

    def _load_code_file(self, file_path: str) -> List[Document]:
        """Parses source code with language awareness."""
        try:
            loader = GenericLoader.from_filesystem(
                file_path, 
                parser=LanguageParser()
            )
            return loader.load()
        except Exception as e:
            self.logger.error(f"Code Loader Error: {e}")
            return []

    def _load_jupyter_notebook(self, file_path: str) -> List[Document]:
        """Loads Jupyter Notebooks including outputs."""
        try:
            loader = NotebookLoader(file_path, include_outputs=True)
            return loader.load()
        except Exception as e:
            self.logger.error(f"Notebook Error: {e}")
            return []

    def _convert_and_load(self, file_path: str) -> List[Document]:
        """Converts Office formats to PDF and then loads them."""
        try:
            pdf_path = self.file_converter.convert_to_pdf(
                file_path, 
                output_file="temp_file", 
                output_dir="./temp_folder/"
            )
            return self._load_pdf(pdf_path)
        except Exception as e:
            self.logger.error(f"Office Conversion Error: {e}")
            return []

    def html_table_to_markdown(self, html_string: str) -> str:
        """Converts HTML table strings to clean Markdown format."""
        soup = BeautifulSoup(html_string, 'html.parser')
        table = soup.find('table')
        if not table: return html_string
        
        rows = []
        for tr in table.find_all('tr'):
            cells = [re.sub(r'\s+', ' ', td.get_text().strip()) for td in tr.find_all(['td', 'th'])]
            rows.append(f"| {' | '.join(cells)} |")
        
        if not rows: return ""
        # Create separator
        num_cols = rows[0].count('|') - 1
        separator = f"| {' | '.join(['---'] * num_cols)} |"
        rows.insert(1, separator)
        return '\n'.join(rows)

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Main entry point. Detects extension and routes to the correct loader.
        """
        source_path = Path(source)
        suffix = source_path.suffix.lower()
        handler = self.document_loaders.get(suffix)
        
        if not handler:
            raise ValueError(f"Unsupported file type: {suffix}")
            
        try:
            # If the handler is a method of this class
            if hasattr(self, handler.__name__) if hasattr(handler, '__name__') else False:
                return handler(str(source_path))
            
            # If the handler is a standard LangChain Loader class
            loader = handler(str(source_path))
            return loader.load()
                
        except Exception as e:
            self.logger.error(f"Final Load Error on {source}: {str(e)}")
            return []