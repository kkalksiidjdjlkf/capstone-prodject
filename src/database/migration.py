"""
Data Migration Script for KREPS RAG System
Migrates data from file-based cache (pickle files) to PostgreSQL hybrid database.
"""

import os
import pickle
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import logging

from .client import DatabaseClient

logger = logging.getLogger(__name__)

class DataMigrator:
    """
    Migrates data from file cache to PostgreSQL database.
    Handles documents, chunks, and maintains FAISS linkages.
    """

    def __init__(self, db_config: Dict[str, Any], cache_dir: str = "./cache"):
        """
        Initialize migrator with database config and cache directory.

        Args:
            db_config: PostgreSQL connection configuration
            cache_dir: Directory containing pickle cache files
        """
        self.db_client = DatabaseClient(db_config)
        self.cache_dir = Path(cache_dir)
        self.migrated_docs = set()
        self.migrated_chunks = set()

    def migrate_all_data(self) -> Dict[str, int]:
        """
        Perform complete data migration from cache to database.

        Returns:
            Dict with migration statistics
        """
        logger.info("Starting complete data migration...")

        stats = {
            'documents_migrated': 0,
            'chunks_migrated': 0,
            'errors': 0
        }

        try:
            # Migrate documents first
            doc_stats = self.migrate_documents()
            stats.update(doc_stats)

            # Then migrate chunks
            chunk_stats = self.migrate_chunks()
            stats.update(chunk_stats)

            # Update system metrics
            self.db_client.update_system_metrics()

            logger.info(f"Migration completed: {stats}")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            stats['errors'] += 1

        return stats

    def migrate_documents(self) -> Dict[str, int]:
        """Migrate document metadata from cache."""
        logger.info("Migrating documents...")

        stats = {'documents_migrated': 0, 'document_errors': 0}

        # Look for document metadata files
        doc_files = list(self.cache_dir.glob("docs_*.pkl"))

        for doc_file in doc_files:
            try:
                with open(doc_file, 'rb') as f:
                    doc_data = pickle.load(f)

                # Extract document info
                for source_path, doc_info in doc_data.items():
                    if source_path in self.migrated_docs:
                        continue

                    # Calculate file hash for deduplication
                    file_hash = self._calculate_file_hash(source_path)

                    # Determine language from file content or name
                    language = self._detect_document_language(source_path)

                    # Create document record
                    doc_id = self.db_client.create_document(
                        file_name=Path(source_path).name,
                        file_path=source_path,
                        file_type=self._get_file_type(source_path),
                        language=language,
                        sensitivity_level="internal",  # Default for now
                        file_hash=file_hash
                    )

                    self.migrated_docs.add(source_path)
                    stats['documents_migrated'] += 1
                    logger.info(f"Migrated document: {source_path}")

            except Exception as e:
                logger.error(f"Failed to migrate document {doc_file}: {e}")
                stats['document_errors'] += 1

        return stats

    def migrate_chunks(self) -> Dict[str, int]:
        """Migrate document chunks from cache."""
        logger.info("Migrating document chunks...")

        stats = {'chunks_migrated': 0, 'chunk_errors': 0}

        # Look for chunk files
        chunk_files = list(self.cache_dir.glob("chunks_*.pkl"))

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)

                # Process chunks
                for source_path, chunks in chunk_data.items():
                    # Find corresponding document
                    doc_record = self._find_document_by_path(source_path)
                    if not doc_record:
                        logger.warning(f"No document record found for {source_path}")
                        continue

                    doc_id = doc_record['id']
                    chunk_count = 0

                    # Process each chunk
                    for i, chunk in enumerate(chunks):
                        if hasattr(chunk, 'metadata') and chunk.metadata.get('source') in self.migrated_chunks:
                            continue

                        # Create chunk record
                        chunk_text = chunk.page_content
                        language = self._detect_chunk_language(chunk_text)

                        # Generate embedding ID for FAISS linkage
                        embedding_id = f"{doc_id}_{i}"

                        chunk_id = self.db_client.create_document_chunk(
                            document_id=doc_id,
                            chunk_text=chunk_text,
                            chunk_index=i,
                            start_char=None,  # Not available in cache
                            end_char=None,
                            language=language,
                            metadata=getattr(chunk, 'metadata', {}),
                            embedding_id=embedding_id,
                            token_count=self._estimate_token_count(chunk_text)
                        )

                        if hasattr(chunk, 'metadata'):
                            chunk.metadata['source'] = source_path

                        self.migrated_chunks.add(source_path)
                        stats['chunks_migrated'] += 1
                        chunk_count += 1

                    # Update document chunk count
                    if chunk_count > 0:
                        self.db_client.update_document_chunk_count(doc_id, chunk_count)

                    logger.info(f"Migrated {chunk_count} chunks for document: {source_path}")

            except Exception as e:
                logger.error(f"Failed to migrate chunks from {chunk_file}: {e}")
                stats['chunk_errors'] += 1

        return stats

    def validate_migration(self) -> Dict[str, Any]:
        """
        Validate that migration was successful.
        Compare cache files with database records.
        """
        logger.info("Validating migration integrity...")

        validation = {
            'cache_documents': 0,
            'db_documents': 0,
            'cache_chunks': 0,
            'db_chunks': 0,
            'discrepancies': []
        }

        # Count documents in cache
        doc_files = list(self.cache_dir.glob("docs_*.pkl"))
        for doc_file in doc_files:
            try:
                with open(doc_file, 'rb') as f:
                    doc_data = pickle.load(f)
                    validation['cache_documents'] += len(doc_data)
            except Exception as e:
                validation['discrepancies'].append(f"Error reading {doc_file}: {e}")

        # Count chunks in cache
        chunk_files = list(self.cache_dir.glob("chunks_*.pkl"))
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    for source_path, chunks in chunk_data.items():
                        validation['cache_chunks'] += len(chunks)
            except Exception as e:
                validation['discrepancies'].append(f"Error reading {chunk_file}: {e}")

        # Get counts from database
        db_counts = self.db_client.get_table_counts()
        validation['db_documents'] = db_counts.get('documents', 0)
        validation['db_chunks'] = db_counts.get('document_chunks', 0)

        # Check for discrepancies
        if validation['cache_documents'] != validation['db_documents']:
            validation['discrepancies'].append(
                f"Document count mismatch: cache={validation['cache_documents']}, db={validation['db_documents']}"
            )

        if validation['cache_chunks'] != validation['db_chunks']:
            validation['discrepancies'].append(
                f"Chunk count mismatch: cache={validation['cache_chunks']}, db={validation['db_chunks']}"
            )

        logger.info(f"Validation completed: {validation}")
        return validation

    def _find_document_by_path(self, source_path: str) -> Dict:
        """Find document record by file path."""
        # This is a simplified implementation
        # In production, you might want to cache document lookups
        try:
            # Query by file_path (this assumes paths are stored correctly)
            with self.db_client.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT * FROM documents WHERE file_path = %s",
                        (source_path,)
                    )
                    result = cursor.fetchone()
                    if result:
                        return {
                            'id': str(result[0]),
                            'file_name': result[1],
                            'file_path': result[2]
                        }
        except Exception as e:
            logger.error(f"Error finding document by path {source_path}: {e}")

        return None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for deduplication."""
        try:
            if os.path.exists(file_path):
                hash_sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
            else:
                # File doesn't exist, use path hash
                return hashlib.sha256(file_path.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()

    def _detect_document_language(self, file_path: str) -> str:
        """Detect document language from file path or content."""
        file_name = Path(file_path).name.lower()

        # Check filename for language indicators
        if 'ko' in file_name or 'korean' in file_name:
            return 'ko'
        elif any(kw in file_name for kw in ['en', 'english', 'matter', 'spec']):
            return 'en'

        # Default to mixed for unknown
        return 'mixed'

    def _detect_chunk_language(self, text: str) -> str:
        """Detect language of text chunk."""
        try:
            from langdetect import detect
            return detect(text[:500])  # Use first 500 chars
        except:
            # Fallback: check for Korean characters
            if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text[:200]):
                return 'ko'
            else:
                return 'en'

    def _get_file_type(self, file_path: str) -> str:
        """Get file type from extension."""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.pdf': 'pdf',
            '.txt': 'txt',
            '.md': 'md',
            '.csv': 'csv'
        }
        return type_map.get(ext, 'unknown')

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

def main():
    """Command-line interface for data migration."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate data from cache to PostgreSQL")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate migration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")

    args = parser.parse_args()

    # Database configuration
    db_config = {
        'url': 'postgresql://rag_user:rag_password@localhost:5432/rag_system'
    }

    migrator = DataMigrator(db_config, args.cache_dir)

    if args.validate_only:
        print("🔍 Validating migration...")
        validation = migrator.validate_migration()
        print("Validation Results:")
        print(f"  Cache documents: {validation['cache_documents']}")
        print(f"  DB documents: {validation['db_documents']}")
        print(f"  Cache chunks: {validation['cache_chunks']}")
        print(f"  DB chunks: {validation['db_chunks']}")
        if validation['discrepancies']:
            print("Discrepancies found:")
            for disc in validation['discrepancies']:
                print(f"  - {disc}")
        else:
            print("✅ No discrepancies found!")

    elif args.dry_run:
        print("🔍 Dry run - analyzing data to be migrated...")
        # Count files that would be migrated
        doc_files = list(Path(args.cache_dir).glob("docs_*.pkl"))
        chunk_files = list(Path(args.cache_dir).glob("chunks_*.pkl"))

        print(f"Documents files to migrate: {len(doc_files)}")
        print(f"Chunk files to migrate: {len(chunk_files)}")

    else:
        print("🚀 Starting data migration...")
        stats = migrator.migrate_all_data()

        print("Migration completed:")
        print(f"  Documents migrated: {stats.get('documents_migrated', 0)}")
        print(f"  Chunks migrated: {stats.get('chunks_migrated', 0)}")
        print(f"  Errors: {stats.get('errors', 0)}")

        if stats.get('errors', 0) == 0:
            print("✅ Migration successful!")
        else:
            print("⚠️  Migration completed with errors. Check logs.")

if __name__ == "__main__":
    main()
