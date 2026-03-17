# RAG System Testing Guide

This document provides comprehensive instructions for testing the RAG (Retrieval-Augmented Generation) system.

## 🧪 Test Suite Overview

The test suite covers all major components of the RAG system:

### **Test Categories:**
- **Unit Tests** (`tests/test_rag_system.py`): Individual component testing
- **API Tests** (`tests/test_api.py`): FastAPI endpoint validation
- **Integration Tests**: End-to-end system validation
- **Security Tests**: Vulnerability and safety checks

### **Components Tested:**
- Database operations (PostgreSQL + file caching)
- BM25 text retrieval
- Vector embeddings (FAISS + HuggingFace)
- Hybrid retrieval fusion
- Document reranking (cross-encoder)
- Response generation (Ollama/LLMs)
- API endpoints and validation
- Docker containerization

## 🚀 Quick Start

### **Run All Tests (Recommended)**
```bash
# Complete test suite with container rebuild
python run_tests.py

# Skip container rebuild (faster)
python run_tests.py --no-rebuild

# Run only unit tests (no containers needed)
python run_tests.py --unit-only
```

### **Run Specific Test Suites**
```bash
# Unit tests only
python -m pytest tests/test_rag_system.py -v

# API tests only
python -m pytest tests/test_api.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

## 🔧 Test Configuration

### **pytest.ini Configuration**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
addopts = -v --tb=short --color=yes
markers =
    unit: Unit tests
    integration: Integration tests
    api: API endpoint tests
    performance: Performance tests
    security: Security tests
```

### **Environment Setup**
- Docker and Docker Compose installed
- Python 3.10+ with required dependencies
- PostgreSQL database (via Docker)
- Sufficient disk space for model downloads

## 📋 Test Scenarios

### **Security Tests**
```bash
# Check PyTorch vulnerability (CVE-2025-32434)
python -c "import torch; print(f'PyTorch {torch.__version__} secure' if float(torch.__version__.split('.')[1]) >= 6 else 'VULNERABLE')"
```

### **API Endpoint Tests**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

### **Performance Tests**
```bash
# Load testing (requires locust or similar)
# Test concurrent requests and response times
```

## 🔍 Test Structure

### **Unit Tests (`test_rag_system.py`)**

#### **Database Tests**
- Connection establishment
- Query logging functionality
- Performance metrics storage
- Hybrid storage operations

#### **Retrieval Tests**
- BM25 document indexing and querying
- Vector embedding generation and storage
- Hybrid fusion retrieval
- Document reranking

#### **Component Tests**
- Cache manager operations
- Reranker model loading and inference
- Response generator integration

### **API Tests (`test_api.py`)**

#### **Endpoint Tests**
- Health check endpoint
- Query processing endpoint
- Error handling scenarios
- Input validation

#### **Validation Tests**
- Query parameter validation
- Response format validation
- CORS and security headers
- Rate limiting (if implemented)

#### **Performance Tests**
- Response time validation
- Concurrent request handling
- Memory usage monitoring

## 🐳 Docker Testing

### **Container Health Checks**
```bash
# Check all containers
docker-compose ps

# View container logs
docker-compose logs backend
docker-compose logs frontend

# Restart specific service
docker-compose restart backend
```

### **Container Testing Commands**
```bash
# Run tests inside backend container
docker-compose exec backend python -m pytest tests/ -v

# Access container shell
docker-compose exec backend bash

# Check container resource usage
docker stats
```

## 📊 Test Reports

### **Generate Coverage Report**
```bash
# HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage summary
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### **Performance Profiling**
```bash
# Profile test execution
python -m pytest tests/ --profile

# Memory profiling
python -m pytest tests/ --memory
```

## 🛠️ Debugging Failed Tests

### **Common Issues**

#### **1. PyTorch Security Error**
```
Error: torch.load vulnerability CVE-2025-32434
```
**Solution:**
```bash
# Update PyTorch version
pip install torch>=2.6.0
# Or rebuild containers
docker-compose build --no-cache backend
```

#### **2. Container Connection Issues**
```
ConnectionError: Backend not responding
```
**Solution:**
```bash
# Check container status
docker-compose ps
# Restart services
docker-compose restart
# Check logs
docker-compose logs backend
```

#### **3. Model Download Failures**
```
HTTPError: Model download failed
```
**Solution:**
```bash
# Clear cache and retry
docker-compose exec backend rm -rf /tmp/*
docker-compose restart backend
```

#### **4. Database Connection Issues**
```
OperationalError: Database connection failed
```
**Solution:**
```bash
# Check PostgreSQL container
docker-compose logs postgres
# Reset database
docker-compose down -v
docker-compose up -d postgres
```

### **Debug Commands**
```bash
# Verbose test output
python -m pytest tests/ -v -s

# Stop on first failure
python -m pytest tests/ -x

# Run specific test
python -m pytest tests/test_rag_system.py::TestDatabaseClient::test_database_connection -v

# Debug with pdb
python -m pytest tests/ --pdb
```

## 📈 Performance Benchmarks

### **Expected Performance**
- **Unit Tests**: < 30 seconds
- **API Tests**: < 2 minutes
- **Integration Tests**: < 5 minutes
- **End-to-End Tests**: < 10 minutes

### **Resource Requirements**
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for model loading
- **Disk**: 10GB+ for models and cache
- **Network**: Stable connection for model downloads

## 🔒 Security Testing

### **Vulnerability Checks**
```bash
# Check for known vulnerabilities
pip audit

# Security linting
bandit -r src/

# Dependency vulnerability scan
safety check
```

### **Input Validation Tests**
- SQL injection prevention
- XSS attack prevention
- Path traversal protection
- File upload validation

## 🎯 CI/CD Integration

### **GitHub Actions Example**
```yaml
name: Test RAG System
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run unit tests
      run: python -m pytest tests/test_rag_system.py --cov=src
    - name: Build and test containers
      run: |
        docker-compose build
        python run_tests.py --no-rebuild
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Docker Testing Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Security Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

## 🆘 Troubleshooting

### **Getting Help**
1. Check test output for specific error messages
2. Review container logs: `docker-compose logs`
3. Verify environment setup
4. Check system resources
5. Review configuration files

### **Common Error Patterns**
- **Import errors**: Check Python path and dependencies
- **Connection errors**: Verify service networking
- **Timeout errors**: Check resource allocation
- **Memory errors**: Monitor system resources

---

**Last Updated**: January 2026
**Test Coverage**: 90%+ (target)
**Security**: CVE-2025-32434 patched
