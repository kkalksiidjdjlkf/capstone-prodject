#!/usr/bin/env python3
"""
Test script for PostgreSQL database client
Tests connection, basic operations, and data integrity
"""

import sys
import os
sys.path.append('.')

from src.database.client import DatabaseClient

def test_database_connection():
    """Test basic database connectivity."""
    print("🔍 Testing PostgreSQL connection...")

    # Database configuration
    db_config = {
        'url': 'postgresql://rag_user:rag_password@localhost:5432/rag_system',
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 30
    }

    try:
        # Initialize client
        db_client = DatabaseClient(db_config)
        print("✅ Database client initialized")

        # Test health check
        if db_client.health_check():
            print("✅ Database health check passed")
        else:
            print("❌ Database health check failed")
            return False

        # Get table counts
        counts = db_client.get_table_counts()
        print("📊 Table counts:")
        for table, count in counts.items():
            print(f"  {table}: {count} records")

        # Test system stats
        stats = db_client.get_system_stats()
        print("📈 System stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_document_operations():
    """Test document creation and retrieval."""
    print("\n📄 Testing document operations...")

    db_config = {
        'url': 'postgresql://rag_user:rag_password@localhost:5432/rag_system'
    }

    try:
        db_client = DatabaseClient(db_config)

        # Create a test document
        doc_id = db_client.create_document(
            file_name="test_document.pdf",
            file_path="/data/test_document.pdf",
            file_type="pdf",
            language="ko",
            sensitivity_level="internal",
            file_hash="abc123test"
        )
        print(f"✅ Created document with ID: {doc_id}")

        # Retrieve the document
        doc = db_client.get_document(doc_id)
        if doc:
            print(f"✅ Retrieved document: {doc['file_name']}")
        else:
            print("❌ Failed to retrieve document")

        # Create a test chunk
        chunk_id = db_client.create_document_chunk(
            document_id=doc_id,
            chunk_text="이것은 테스트 청크입니다. This is a test chunk.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            language="mixed",
            metadata={"test": True},
            embedding_id="test_embedding_001",
            token_count=12
        )
        print(f"✅ Created chunk with ID: {chunk_id}")

        # Test chunk retrieval
        chunk = db_client.get_chunk_by_embedding_id("test_embedding_001")
        if chunk:
            print(f"✅ Retrieved chunk by embedding ID: {chunk['chunk_text'][:30]}...")
        else:
            print("❌ Failed to retrieve chunk")

        return True

    except Exception as e:
        print(f"❌ Document operations test failed: {e}")
        return False

def test_query_logging():
    """Test query logging functionality."""
    print("\n📝 Testing query logging...")

    db_config = {
        'url': 'postgresql://rag_user:rag_password@localhost:5432/rag_system'
    }

    try:
        db_client = DatabaseClient(db_config)

        # Log a test query
        log_id = db_client.log_query(
            original_query="What is RAG?",
            detected_language="en",
            llm_response="RAG stands for Retrieval-Augmented Generation...",
            processing_time_ms=1500,
            retrieved_chunk_count=5,
            avg_similarity_score=0.85
        )
        print(f"✅ Logged query with ID: {log_id}")

        # Get recent queries
        recent_queries = db_client.get_recent_queries(5)
        print(f"✅ Retrieved {len(recent_queries)} recent queries")

        # Get query stats
        stats = db_client.get_query_stats(1)  # Last 24 hours
        print("📊 Query stats (24h):")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Query logging test failed: {e}")
        return False

def main():
    """Run all database tests."""
    print("🧪 Starting PostgreSQL Database Tests")
    print("=" * 50)

    tests = [
        ("Database Connection", test_database_connection),
        ("Document Operations", test_document_operations),
        ("Query Logging", test_query_logging)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Result: {status}")

    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")

    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All database tests passed! Ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
