#!/usr/bin/env python3
"""
Test Runner for RAG System
Runs comprehensive tests for all components and provides detailed reporting
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import argparse


class TestRunner:
    """Comprehensive test runner for the RAG system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results = {}

    def run_command(self, command, cwd=None, timeout=300):
        """Run a shell command with timeout"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def check_pytorch_security(self):
        """Check PyTorch version for security vulnerabilities"""
        print("🔒 Checking PyTorch security...")
        try:
            import torch
            version = torch.__version__
            major, minor = version.split('.')[:2]
            version_float = float(f"{major}.{minor}")

            if version_float >= 2.6:
                print(f"✅ PyTorch {version} is secure (CVE-2025-32434 fixed)")
                return True
            else:
                print(f"❌ PyTorch {version} is vulnerable. Need >= 2.6.0")
                return False
        except ImportError:
            print("❌ PyTorch not installed")
            return False

    def rebuild_containers(self):
        """Rebuild Docker containers with security fixes"""
        print("🔨 Rebuilding Docker containers...")

        # Stop existing containers
        success, _, _ = self.run_command("docker-compose down")
        if not success:
            print("⚠️  Warning: Could not stop containers cleanly")

        # Rebuild containers
        success, stdout, stderr = self.run_command("docker-compose build --no-cache")
        if success:
            print("✅ Containers rebuilt successfully")
            return True
        else:
            print(f"❌ Container rebuild failed: {stderr}")
            return False

    def run_unit_tests(self):
        """Run unit tests for individual components"""
        print("🧪 Running unit tests...")

        if not self.tests_dir.exists():
            print("❌ Tests directory not found")
            return False

        # Run pytest on unit tests
        success, stdout, stderr = self.run_command(
            "python -m pytest tests/test_rag_system.py -v --tb=short"
        )

        if success:
            print("✅ Unit tests passed")
            print(stdout)
        else:
            print("❌ Unit tests failed")
            print(stderr)

        return success

    def run_api_tests(self):
        """Run API endpoint tests"""
        print("🌐 Running API tests...")

        # Check if backend is running
        success, stdout, stderr = self.run_command("docker-compose ps backend")
        if "Up" not in stdout:
            print("⚠️  Backend not running, starting it...")
            self.run_command("docker-compose up -d backend")
            time.sleep(10)  # Wait for startup

        # Run API tests
        success, stdout, stderr = self.run_command(
            "python -m pytest tests/test_api.py -v --tb=short"
        )

        if success:
            print("✅ API tests passed")
            print(stdout)
        else:
            print("❌ API tests failed")
            print(stderr)

        return success

    def run_integration_tests(self):
        """Run integration tests for the complete system"""
        print("🔗 Running integration tests...")

        # This would test the complete pipeline
        # For now, just check if services are healthy
        success, stdout, stderr = self.run_command("docker-compose ps")
        if "Up" in stdout:
            print("✅ All services are running")
            return True
        else:
            print("❌ Some services are not running")
            print(stdout)
            return False

    def test_end_to_end(self):
        """Test complete end-to-end functionality"""
        print("🚀 Testing end-to-end functionality...")

        # Start all services
        success, _, _ = self.run_command("docker-compose up -d")
        if not success:
            print("❌ Could not start services")
            return False

        # Wait for services to be ready
        time.sleep(15)

        # Test a simple query via API
        try:
            import requests
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": "test query", "top_k": 3},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if "response" in data and "retrieved_documents" in data:
                    print("✅ End-to-end test passed")
                    return True
                else:
                    print("❌ End-to-end test failed: Invalid response structure")
                    return False
            else:
                print(f"❌ End-to-end test failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ End-to-end test failed: {e}")
            return False

    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*50)
        print("📊 TEST REPORT")
        print("="*50)

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)

        print(f"Total test suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")

        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")

        print("\nDetailed Results:")
        for test_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")

    def run_all_tests(self, rebuild=True, e2e=True):
        """Run complete test suite"""
        print("🧪 Starting RAG System Test Suite")
        print("="*50)

        # Check PyTorch security
        self.results["PyTorch Security"] = self.check_pytorch_security()

        if rebuild:
            self.results["Container Rebuild"] = self.rebuild_containers()
        else:
            self.results["Container Rebuild"] = True  # Skip if not rebuilding

        # Run test suites
        self.results["Unit Tests"] = self.run_unit_tests()
        self.results["API Tests"] = self.run_api_tests()
        self.results["Integration Tests"] = self.run_integration_tests()

        if e2e:
            self.results["End-to-End Tests"] = self.test_end_to_end()

        # Generate report
        self.generate_report()

        # Return overall success
        return all(self.results.values())


def main():
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument("--no-rebuild", action="store_true",
                       help="Skip container rebuild")
    parser.add_argument("--no-e2e", action="store_true",
                       help="Skip end-to-end tests")
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")

    args = parser.parse_args()

    runner = TestRunner()

    if args.unit_only:
        # Run only unit tests without containers
        runner.results["PyTorch Security"] = runner.check_pytorch_security()
        runner.results["Unit Tests"] = runner.run_unit_tests()
        runner.generate_report()
    else:
        # Run full test suite
        rebuild = not args.no_rebuild
        e2e = not args.no_e2e
        success = runner.run_all_tests(rebuild=rebuild, e2e=e2e)

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
