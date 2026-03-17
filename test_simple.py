#!/usr/bin/env python3
"""
Simple test to verify the RAG system components work
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from database.client import DatabaseClient
        print("✅ DatabaseClient import successful")

        from cache.cache import CacheManager
        print("✅ CacheManager import successful")

        from retrieval.vector_retriever import VectorRetriever
        print("✅ VectorRetriever import successful")

        from retrieval.hybrid_retriever import HybridRetriever
        print("✅ HybridRetriever import successful")

        from models.reranker import Reranker
        print("✅ Reranker import successful")

        from response.response_generator import ResponseGenerator
        print("✅ ResponseGenerator import successful")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytorch_security():
    """Test PyTorch security"""
    try:
        import torch
        version = torch.__version__
        major, minor = version.split('.')[:2]
        version_tuple = (int(major), int(minor))

        if version_tuple >= (2, 6):
            print(f"✅ PyTorch {version} is secure (CVE-2025-32434 fixed)")
            return True
        else:
            print(f"❌ PyTorch {version} is vulnerable. Need >= 2.6.0")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_cache_manager():
    """Test cache manager basic functionality"""
    try:
        from cache.cache import CacheManager
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(temp_dir)

            # Test basic operations
            test_data = {"key": "value", "number": 42}

            # Set and get
            cache.set("test_key", test_data)
            retrieved = cache.get("test_key")

            if retrieved == test_data:
                print("✅ Cache manager working correctly")
                return True
            else:
                print("❌ Cache retrieval failed")
                return False

    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return False

def test_database_connection():
    """Test database connection (mock)"""
    try:
        from database.client import DatabaseClient

        # Test with mock config (won't actually connect)
        config = {'url': 'postgresql://test:test@localhost:5432/test'}
        db_client = DatabaseClient(config)

        print("✅ DatabaseClient initialization successful")
        return True

    except Exception as e:
        print(f"❌ Database client test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Simple RAG System Tests")
    print("=" * 50)

    tests = [
        ("PyTorch Security", test_pytorch_security),
        ("Module Imports", test_imports),
        ("Cache Manager", test_cache_manager),
        ("Database Client", test_database_connection),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
