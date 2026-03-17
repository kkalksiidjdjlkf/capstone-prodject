"""
PostgreSQL Database Client for KREPS RAG System
Handles structured data storage and retrieval in hybrid PostgreSQL + FAISS architecture.
"""

import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    import psycopg2.pool
except ImportError:
    raise ImportError("psycopg2 is required for PostgreSQL database operations. Install with: pip install psycopg2-binary")

logger = logging.getLogger(__name__)

class DatabaseClient:
    """
    PostgreSQL client for hybrid RAG system.
    Handles structured data while FAISS manages vectors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database client with connection parameters.

        Args:
            config: Database configuration containing:
                - url: PostgreSQL connection URL
                - pool_size: Connection pool size
                - max_overflow: Max overflow connections
                - pool_timeout: Connection timeout
        """
        self.config = config
        self.connection_pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool for efficient database access."""
        try:
            # Parse connection URL
            url = self.config.get('url', 'postgresql://rag_user:rag_password@localhost:5432/rag_system')

            # Parse URL components for connection pool
            # Format: postgresql://user:password@host:port/database
            if url.startswith('postgresql://'):
                url_parts = url.replace('postgresql://', '').split('@')
                if len(url_parts) == 2:
                    credentials = url_parts[0].split(':')
                    host_port_db = url_parts[1].split('/')

                    if len(credentials) == 2 and len(host_port_db) >= 1:
                        user, password = credentials
                        host_port = host_port_db[0].split(':')
                        database = host_port_db[1] if len(host_port_db) > 1 else 'rag_system'

                        host = host_port[0]
                        port = int(host_port[1]) if len(host_port) > 1 else 5432

                        # Initialize connection pool
                        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                            minconn=self.config.get('pool_size', 5),      # Minimum connections
                            maxconn=self.config.get('max_connections', 20), # Maximum connections
                            host=host,
                            port=port,
                            database=database,
                            user=user,
                            password=password,
                            connect_timeout=self.config.get('timeout', 10)
                        )

                        logger.info(f"Database connection pool initialized: min={self.config.get('pool_size', 5)}, max={self.config.get('max_connections', 20)}")
                        return

            # Fallback to URL-based connections if parsing fails
            logger.warning("Could not parse database URL for connection pool, falling back to URL connections")
            self.connection_url = url
            logger.info("Database client initialized with URL connections")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with pooling support."""
        conn = None
        try:
            if self.connection_pool:
                # Use connection pool (much faster!)
                conn = self.connection_pool.getconn()
            else:
                # Fallback to direct connection
                conn = psycopg2.connect(self.connection_url)

            conn.autocommit = False  # Use transactions
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                if self.connection_pool:
                    # Return connection to pool
                    self.connection_pool.putconn(conn)
                else:
                    # Close direct connection
                    conn.close()

    # ===========================================
    # DOCUMENTS TABLE OPERATIONS
    # ===========================================

    def create_document(self, file_name: str, file_path: str, file_type: str,
                       language: str, sensitivity_level: str, file_hash: str) -> str:
        """
        Create a new document record.

        Args:
            file_name: Name of the uploaded file
            file_path: Path to the file
            file_type: File type (pdf, txt, md, csv)
            language: Document language (ko, en, mixed)
            sensitivity_level: Security level (public, internal, confidential)
            file_hash: SHA256 hash for deduplication

        Returns:
            UUID of created document
        """
        query = """
        INSERT INTO documents (file_name, file_path, file_type, language, sensitivity_level, file_hash)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (file_name, file_path, file_type, language, sensitivity_level, file_hash))
                doc_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Created document record: {doc_id} for {file_name}")
                return str(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        query = "SELECT * FROM documents WHERE id = %s"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (doc_id,))
                result = cursor.fetchone()
                return dict(result) if result else None

    def update_document_chunk_count(self, doc_id: str, chunk_count: int):
        """Update chunk count for a document."""
        query = "UPDATE documents SET chunk_count = %s, is_indexed = TRUE WHERE id = %s"

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (chunk_count, doc_id))
                conn.commit()
                logger.info(f"Updated chunk count for document {doc_id}: {chunk_count}")

    def get_documents_by_language(self, language: str) -> List[Dict]:
        """Get all documents in a specific language."""
        query = "SELECT * FROM documents WHERE language = %s ORDER BY created_at DESC"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (language,))
                return [dict(row) for row in cursor.fetchall()]

    # ===========================================
    # DOCUMENT CHUNKS TABLE OPERATIONS
    # ===========================================

    def create_document_chunk(self, document_id: str, chunk_text: str, chunk_index: int,
                            start_char: Optional[int], end_char: Optional[int],
                            language: str, metadata: Optional[Dict], embedding_id: str,
                            token_count: Optional[int]) -> str:
        """
        Create a new document chunk record.

        Args:
            document_id: Parent document UUID
            chunk_text: The actual text content (max 2000 chars)
            chunk_index: Position within document
            start_char/end_char: Character offsets
            language: Chunk language (ko, en, mixed)
            metadata: Additional metadata as JSON
            embedding_id: Link to FAISS vector
            token_count: Number of tokens in chunk

        Returns:
            UUID of created chunk
        """
        query = """
        INSERT INTO document_chunks (
            document_id, chunk_text, chunk_index, start_char, end_char,
            language, metadata_json, embedding_id, token_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    document_id, chunk_text, chunk_index, start_char, end_char,
                    language, Json(metadata) if metadata else None, embedding_id, token_count
                ))
                chunk_id = cursor.fetchone()[0]
                conn.commit()
                return str(chunk_id)

    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        query = "SELECT * FROM document_chunks WHERE document_id = %s ORDER BY chunk_index"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (document_id,))
                return [dict(row) for row in cursor.fetchall()]

    def get_chunk_by_embedding_id(self, embedding_id: str) -> Optional[Dict]:
        """Get chunk by its FAISS embedding ID."""
        query = "SELECT * FROM document_chunks WHERE embedding_id = %s"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (embedding_id,))
                result = cursor.fetchone()
                return dict(result) if result else None

    def get_chunks_by_language(self, language: str, limit: int = 100) -> List[Dict]:
        """Get chunks by language with optional limit."""
        query = "SELECT * FROM document_chunks WHERE language = %s ORDER BY created_at DESC LIMIT %s"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (language, limit))
                return [dict(row) for row in cursor.fetchall()]

    # ===========================================
    # QUERY LOGS TABLE OPERATIONS
    # ===========================================

    def log_query(self, original_query: str, detected_language: str,
                  llm_response: str, processing_time_ms: int,
                  retrieved_chunk_count: int, avg_similarity_score: float,
                  user_id: Optional[str] = None) -> str:
        """
        Log a query interaction for audit and analytics.

        Args:
            original_query: The user's original question
            detected_language: Detected language (ko, en)
            llm_response: Generated response
            processing_time_ms: Time taken to process
            retrieved_chunk_count: Number of chunks retrieved
            avg_similarity_score: Average similarity score
            user_id: Optional user identifier

        Returns:
            UUID of created log entry
        """
        query = """
        INSERT INTO query_logs (
            user_id, original_query, detected_language, llm_response,
            processing_time_ms, retrieved_chunk_count, avg_similarity_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    user_id, original_query, detected_language, llm_response,
                    processing_time_ms, retrieved_chunk_count, avg_similarity_score
                ))
                log_id = cursor.fetchone()[0]
                conn.commit()
                return str(log_id)

    def get_recent_queries(self, limit: int = 50) -> List[Dict]:
        """Get recent query logs for analytics."""
        query = """
        SELECT * FROM query_logs
        ORDER BY created_at DESC
        LIMIT %s
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (limit,))
                return [dict(row) for row in cursor.fetchall()]

    def get_query_stats(self, days: int = 7) -> Dict:
        """Get query statistics for analytics dashboard."""
        query = """
        SELECT
            COUNT(*) as total_queries,
            AVG(processing_time_ms) as avg_processing_time,
            AVG(avg_similarity_score) as avg_similarity,
            COUNT(CASE WHEN detected_language = 'ko' THEN 1 END) as korean_queries,
            COUNT(CASE WHEN detected_language = 'en' THEN 1 END) as english_queries
        FROM query_logs
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (days,))
                result = cursor.fetchone()
                return dict(result) if result else {}

    # ===========================================
    # SYSTEM METRICS & ANALYTICS
    # ===========================================

    def update_system_metrics(self):
        """Update system-wide metrics."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT update_system_metrics()")
                conn.commit()

    def get_system_stats(self) -> Dict:
        """Get current system statistics."""
        stats = {}

        # Document stats
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total documents
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                stats['total_documents'] = cursor.fetchone()['total']

                # Total chunks
                cursor.execute("SELECT COUNT(*) as total FROM document_chunks")
                stats['total_chunks'] = cursor.fetchone()['total']

                # Language breakdown
                cursor.execute("""
                    SELECT language, COUNT(*) as count
                    FROM documents
                    GROUP BY language
                """)
                stats['documents_by_language'] = {row['language']: row['count'] for row in cursor.fetchall()}

                # Recent queries (last 24h)
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM query_logs
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                stats['queries_24h'] = cursor.fetchone()['count']

        return stats

    # ===========================================
    # UTILITY METHODS
    # ===========================================

    def health_check(self) -> bool:
        """Check database connectivity and basic operations."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all main tables."""
        tables = ['documents', 'document_chunks', 'query_logs', 'users', 'audit_logs']
        counts = {}

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]

        return counts

    def cleanup_old_data(self, days: int = 90):
        """Clean up old query logs and temporary data."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Archive old query logs (in production, move to archive table)
                cursor.execute("""
                    DELETE FROM query_logs
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """, (days,))
                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old query log entries")
                return deleted_count

    def __del__(self):
        """Cleanup connection pool on destruction."""
        if self.connection_pool:
            self.connection_pool.closeall()
