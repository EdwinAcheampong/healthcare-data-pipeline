#!/usr/bin/env python3
"""
Test script for Phase 3 API endpoints.

This script tests all the major endpoints of the healthcare data pipeline API
to ensure they are working correctly.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any


class Phase3APITester:
    """Test class for Phase 3 API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, response_time: float, details: str = ""):
        """Log test results."""
        result = {
            "test": test_name,
            "success": success,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.3f}s) - {details}")
    
    def test_health_endpoints(self):
        """Test health check endpoints."""
        print("\n🏥 Testing Health Endpoints...")
        
        # Test root endpoint
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Root Endpoint", True, response_time, f"Version: {data.get('version', 'unknown')}")
            else:
                self.log_test("Root Endpoint", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Root Endpoint", False, time.time() - start_time, str(e))
        
        # Test health check
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/health")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Check", True, response_time, f"Status: {data.get('status', 'unknown')}")
            else:
                self.log_test("Health Check", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Health Check", False, time.time() - start_time, str(e))
        
        # Test detailed health
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Detailed Health", True, response_time, f"Environment: {data.get('environment', 'unknown')}")
            else:
                self.log_test("Detailed Health", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Detailed Health", False, time.time() - start_time, str(e))
    
    def test_optimization_endpoints(self):
        """Test optimization endpoints."""
        print("\n🔧 Testing Optimization Endpoints...")
        
        # Test optimization strategies
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/optimize/strategies")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                strategies_count = len(data.get('strategies', []))
                self.log_test("Get Optimization Strategies", True, response_time, f"Found {strategies_count} strategies")
            else:
                self.log_test("Get Optimization Strategies", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Get Optimization Strategies", False, time.time() - start_time, str(e))
        
        # Test workload optimization
        optimization_request = {
            "current_patients": 50,
            "current_staff": 20,
            "department": "emergency",
            "shift_hours": 8,
            "optimization_strategy": "ppo",
            "constraints": {"max_staff": 25},
            "target_metrics": ["efficiency", "patient_satisfaction"]
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/optimize/workload",
                json=optimization_request
            )
            response_time = time.time() - start_time
            
            if response.status_code in [200, 202]:
                data = response.json()
                optimization_id = data.get('optimization_id', 'unknown')
                self.log_test("Workload Optimization", True, response_time, f"Job ID: {optimization_id}")
            else:
                self.log_test("Workload Optimization", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Workload Optimization", False, time.time() - start_time, str(e))
    
    def test_prediction_endpoints(self):
        """Test prediction endpoints."""
        print("\n🔮 Testing Prediction Endpoints...")
        
        # Test available models
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/predict/models")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                models_count = len(data.get('models', []))
                self.log_test("Get Available Models", True, response_time, f"Found {models_count} models")
            else:
                self.log_test("Get Available Models", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Get Available Models", False, time.time() - start_time, str(e))
        
        # Test workload prediction
        prediction_request = {
            "model_type": "baseline",
            "input_data": {
                "patient_count": 45,
                "time_of_day": 14,
                "day_of_week": 2,
                "department": "emergency"
            },
            "prediction_horizon": 24,
            "confidence_level": 0.95
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/predict/workload",
                json=prediction_request
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                prediction_id = data.get('prediction_id', 'unknown')
                self.log_test("Workload Prediction", True, response_time, f"Prediction ID: {prediction_id}")
            else:
                self.log_test("Workload Prediction", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Workload Prediction", False, time.time() - start_time, str(e))
        
        # Test model performance
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/predict/performance")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                accuracy = data.get('overall_performance', {}).get('prediction_accuracy', 0)
                self.log_test("Model Performance", True, response_time, f"Accuracy: {accuracy:.2%}")
            else:
                self.log_test("Model Performance", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Model Performance", False, time.time() - start_time, str(e))
    
    def test_monitoring_endpoints(self):
        """Test monitoring endpoints."""
        print("\n📊 Testing Monitoring Endpoints...")
        
        # Test system metrics
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/monitoring/metrics")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("System Metrics", True, response_time, "Metrics retrieved successfully")
            else:
                self.log_test("System Metrics", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("System Metrics", False, time.time() - start_time, str(e))
        
        # Test API performance
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/monitoring/metrics/api")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                avg_response_time = data.get('average_response_time', 0)
                self.log_test("API Performance", True, response_time, f"Avg response: {avg_response_time:.3f}s")
            else:
                self.log_test("API Performance", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("API Performance", False, time.time() - start_time, str(e))
        
        # Test dashboard data
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/api/v1/monitoring/dashboard")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                system_status = data.get('summary', {}).get('system_status', 'unknown')
                self.log_test("Dashboard Data", True, response_time, f"System status: {system_status}")
            else:
                self.log_test("Dashboard Data", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Dashboard Data", False, time.time() - start_time, str(e))
    
    def test_api_documentation(self):
        """Test API documentation endpoints."""
        print("\n📚 Testing API Documentation...")
        
        # Test OpenAPI docs
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/docs")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("API Documentation", True, response_time, "Swagger UI accessible")
            else:
                self.log_test("API Documentation", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("API Documentation", False, time.time() - start_time, str(e))
        
        # Test OpenAPI JSON
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/openapi.json")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                endpoints_count = len(data.get('paths', {}))
                self.log_test("OpenAPI Schema", True, response_time, f"Found {endpoints_count} endpoints")
            else:
                self.log_test("OpenAPI Schema", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("OpenAPI Schema", False, time.time() - start_time, str(e))
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Phase 3 API Tests...")
        print(f"Testing API at: {self.base_url}")
        print("=" * 60)
        
        # Run all test suites
        self.test_health_endpoints()
        self.test_optimization_endpoints()
        self.test_prediction_endpoints()
        self.test_monitoring_endpoints()
        self.test_api_documentation()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['details']}")
        
        # Calculate average response time
        avg_response_time = sum(r['response_time'] for r in self.test_results) / total_tests
        print(f"\n⏱️  Average Response Time: {avg_response_time:.3f}s")
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase3_api_test_results_{timestamp}.json"
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": self.test_results,
            "summary": {
                "total": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r['success']),
                "failed": sum(1 for r in self.test_results if not r['success']),
                "avg_response_time": sum(r['response_time'] for r in self.test_results) / len(self.test_results)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filename}")


def main():
    """Main function to run the API tests."""
    import sys
    
    # Get base URL from command line argument or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Create tester and run tests
    tester = Phase3APITester(base_url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
