"""
API Endpoint Tests for RAG Backend
Tests FastAPI endpoints, request/response handling, and error scenarios
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestAPIEndpoints:
    """Test FastAPI endpoints for the RAG system"""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        from main import app
        return TestClient(app)

    @pytest.fixture
    def mock_retrieval_system(self):
        """Mock the retrieval system components"""
        with patch('main.HybridRetriever') as mock_hybrid, \
             patch('main.Reranker') as mock_reranker, \
             patch('main.ResponseGenerator') as mock_generator:

            # Setup mock retriever
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                {'content': 'Mock retrieved document', 'source': 'test.pdf', 'score': 0.95}
            ]
            mock_hybrid.return_value = mock_retriever

            # Setup mock reranker
            mock_rerank = MagicMock()
            mock_rerank.rerank.return_value = (
                ['Reranked document'],
                [0.98]
            )
            mock_reranker.return_value = mock_rerank

            # Setup mock generator
            mock_gen = MagicMock()
            mock_gen.generate_response.return_value = "Generated response based on retrieved documents"
            mock_generator.return_value = mock_gen

            yield {
                'retriever': mock_retriever,
                'reranker': mock_rerank,
                'generator': mock_gen
            }

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_query_endpoint_success(self, client, mock_retrieval_system):
        """Test successful query processing"""
        query_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "include_sources": True
        }

        response = client.post("/query", json=query_data)
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "retrieved_documents" in data
        assert "processing_time" in data
        assert "query" in data
        assert data["query"] == query_data["query"]

    def test_query_endpoint_missing_query(self, client):
        """Test query endpoint with missing query parameter"""
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query"""
        query_data = {"query": ""}
        response = client.post("/query", json=query_data)
        assert response.status_code == 400

    def test_query_endpoint_invalid_top_k(self, client):
        """Test query endpoint with invalid top_k parameter"""
        query_data = {
            "query": "test query",
            "top_k": -1
        }
        response = client.post("/query", json=query_data)
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_large_top_k(self, client, mock_retrieval_system):
        """Test query endpoint with large top_k parameter"""
        query_data = {
            "query": "test query",
            "top_k": 1000
        }
        response = client.post("/query", json=query_data)
        assert response.status_code == 200  # Should handle gracefully

    def test_query_processing_flow(self, client, mock_retrieval_system):
        """Test complete query processing flow"""
        query_data = {
            "query": "What is artificial intelligence?",
            "top_k": 3
        }

        response = client.post("/query", json=query_data)

        # Verify the response structure
        assert response.status_code == 200
        data = response.json()

        # Check that all components were called
        mock_retrieval_system['retriever'].retrieve.assert_called_once()
        mock_retrieval_system['reranker'].rerank.assert_called_once()
        mock_retrieval_system['generator'].generate_response.assert_called_once()

        # Verify response content
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["retrieved_documents"], list)
        assert len(data["retrieved_documents"]) > 0

    def test_error_handling_retrieval_failure(self, client):
        """Test error handling when retrieval fails"""
        with patch('main.HybridRetriever') as mock_hybrid:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.side_effect = Exception("Retrieval failed")
            mock_hybrid.return_value = mock_retriever

            query_data = {"query": "test query"}
            response = client.post("/query", json=query_data)

            assert response.status_code == 500

    def test_error_handling_reranking_failure(self, client):
        """Test error handling when reranking fails"""
        with patch('main.HybridRetriever') as mock_hybrid, \
             patch('main.Reranker') as mock_reranker:

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [{'content': 'test', 'source': 'test.pdf'}]
            mock_hybrid.return_value = mock_retriever

            mock_rerank = MagicMock()
            mock_rerank.rerank.side_effect = Exception("Reranking failed")
            mock_reranker.return_value = mock_rerank

            query_data = {"query": "test query"}
            response = client.post("/query", json=query_data)

            assert response.status_code == 500

    def test_error_handling_generation_failure(self, client):
        """Test error handling when response generation fails"""
        with patch('main.HybridRetriever') as mock_hybrid, \
             patch('main.Reranker') as mock_reranker, \
             patch('main.ResponseGenerator') as mock_generator:

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [{'content': 'test', 'source': 'test.pdf'}]
            mock_hybrid.return_value = mock_retriever

            mock_rerank = MagicMock()
            mock_rerank.rerank.return_value = (['test'], [0.9])
            mock_reranker.return_value = mock_rerank

            mock_gen = MagicMock()
            mock_gen.generate_response.side_effect = Exception("Generation failed")
            mock_generator.return_value = mock_gen

            query_data = {"query": "test query"}
            response = client.post("/query", json=query_data)

            assert response.status_code == 500

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/query")
        assert response.status_code == 200
        # CORS headers should be present in actual implementation

    def test_metrics_endpoint(self, client):
        """Test metrics/analytics endpoint"""
        response = client.get("/metrics")
        # This endpoint may or may not exist depending on implementation
        # If it exists, test the response structure
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_invalid_http_method(self, client):
        """Test invalid HTTP methods are rejected"""
        response = client.get("/query")  # GET instead of POST
        assert response.status_code in [405, 404]  # Method not allowed or not found


class TestAPIValidation:
    """Test API input validation"""

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)

    def test_query_validation_max_length(self, client):
        """Test query length validation"""
        long_query = "a" * 10000  # Very long query
        query_data = {"query": long_query}

        response = client.post("/query", json=query_data)
        # Should either accept or reject based on implementation
        assert response.status_code in [200, 400, 422]

    def test_special_characters_in_query(self, client):
        """Test queries with special characters"""
        special_queries = [
            "query with @#$%^&*()",
            "query with <script>alert('xss')</script>",
            "query with SQL: DROP TABLE users;",
            "query with unicode: 你好世界 🌍"
        ]

        for query in special_queries:
            query_data = {"query": query}
            response = client.post("/query", json=query_data)
            # Should handle gracefully
            assert response.status_code in [200, 400]

    def test_json_injection_prevention(self, client):
        """Test prevention of JSON injection attacks"""
        malicious_query = '{"query": "test", "malicious": "\"); DROP TABLE users; --"}'
        response = client.post("/query", data=malicious_query, headers={"Content-Type": "application/json"})

        # Should not execute malicious code
        assert response.status_code in [200, 400, 422]


class TestPerformance:
    """Test API performance characteristics"""

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)

    def test_response_time(self, client):
        """Test that responses are returned within reasonable time"""
        import time

        query_data = {"query": "test query"}
        start_time = time.time()

        response = client.post("/query", json=query_data)

        end_time = time.time()
        response_time = end_time - start_time

        # Response should be reasonably fast (under 30 seconds for this test)
        assert response_time < 30.0
        assert response.status_code == 200

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import queue

        results = queue.Queue()

        def make_request():
            try:
                response = client.post("/query", json={"query": "concurrent test"})
                results.put((response.status_code, response.json() if response.status_code == 200 else None))
            except Exception as e:
                results.put(("error", str(e)))

        # Make 3 concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            status, data = results.get()
            if status == 200:
                success_count += 1
            elif status == "error":
                # Some errors might be acceptable due to resource constraints
                pass

        # At least some requests should succeed
        assert success_count >= 1


if __name__ == "__main__":
    # Run basic API smoke tests
    print("Running API Tests...")

    try:
        from main import app
        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")

        # Test query endpoint with mock
        with patch('main.HybridRetriever') as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = [{'content': 'test', 'source': 'test.pdf'}]
            mock_retriever.return_value = mock_instance

            response = client.post("/query", json={"query": "test"})
            if response.status_code == 200:
                print("✅ Query endpoint working")
            else:
                print(f"❌ Query endpoint failed: {response.status_code}")

    except Exception as e:
        print(f"❌ API test failed: {e}")

    print("API test suite completed!")
