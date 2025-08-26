#!/usr/bin/env python3
"""
Phase 3 API Startup Script

This script provides an easy way to start the healthcare data pipeline API
with proper configuration, health checks, and monitoring.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Optional


class Phase3Startup:
    """Phase 3 API startup manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.api_url = "http://localhost:8000"
        self.health_url = f"{self.api_url}/health"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            print("❌ Python 3.11+ is required")
            return False
        
        # Check if requirements are installed
        try:
            import fastapi
            import uvicorn
            import psutil
            print("✅ Python dependencies are installed")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Run: pip install -r requirements.txt")
            return False
        
        # Check if .env file exists
        env_file = self.project_root / ".env"
        if not env_file.exists():
            print("⚠️  .env file not found, using defaults")
        
        print("✅ Prerequisites check passed")
        return True
    
    def setup_environment(self):
        """Setup environment variables."""
        print("⚙️  Setting up environment...")
        
        # Set default environment variables
        os.environ.setdefault("ENVIRONMENT", "development")
        os.environ.setdefault("LOG_LEVEL", "INFO")
        os.environ.setdefault("API_HOST", "0.0.0.0")
        os.environ.setdefault("API_PORT", "8000")
        
        # Load from .env file if it exists
        env_file = self.project_root / ".env"
        if env_file.exists():
            print("📄 Loading environment from .env file")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key] = value
        
        print("✅ Environment setup complete")
    
    def check_dependencies(self) -> bool:
        """Check if external dependencies are available."""
        print("🔗 Checking external dependencies...")
        
        # Check if Docker is running (for database services)
        try:
            result = subprocess.run(
                ["docker", "ps"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                print("✅ Docker is running")
                return self.check_docker_services()
            else:
                print("⚠️  Docker not available, using local services")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Docker not available, using local services")
            return True
    
    def check_docker_services(self) -> bool:
        """Check if Docker services are running."""
        try:
            # Check if services are running
            result = subprocess.run(
                ["docker-compose", "ps", "--services", "--filter", "status=running"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                running_services = result.stdout.strip().split('\n')
                if running_services and running_services[0]:
                    print(f"✅ Docker services running: {', '.join(running_services)}")
                    return True
                else:
                    print("⚠️  No Docker services running")
                    return self.start_docker_services()
            else:
                print("⚠️  Could not check Docker services")
                return True
                
        except Exception as e:
            print(f"⚠️  Docker service check failed: {e}")
            return True
    
    def start_docker_services(self) -> bool:
        """Start Docker services if needed."""
        print("🐳 Starting Docker services...")
        
        try:
            # Start services in background
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.project_root,
                check=True
            )
            
            # Wait for services to be ready
            print("⏳ Waiting for services to be ready...")
            time.sleep(10)
            
            # Check if services are running
            result = subprocess.run(
                ["docker-compose", "ps", "--services", "--filter", "status=running"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                running_services = result.stdout.strip().split('\n')
                if running_services and running_services[0]:
                    print(f"✅ Docker services started: {', '.join(running_services)}")
                    return True
            
            print("❌ Failed to start Docker services")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Docker services: {e}")
            return False
    
    def start_api(self) -> Optional[subprocess.Popen]:
        """Start the API server."""
        print("🚀 Starting API server...")
        
        try:
            # Change to project root
            os.chdir(self.project_root)
            
            # Start the API
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            time.sleep(3)
            
            if process.poll() is None:
                print("✅ API server started successfully")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ API server failed to start:")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return None
                
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return None
    
    def wait_for_api(self, timeout: int = 30) -> bool:
        """Wait for API to be ready."""
        print(f"⏳ Waiting for API to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code == 200:
                    print("✅ API is ready!")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(1)
        
        print("❌ API failed to become ready within timeout")
        return False
    
    def run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        print("🏥 Running health checks...")
        
        try:
            # Basic health check
            response = requests.get(self.health_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Health check failed: {response.status_code}")
                return False
            
            data = response.json()
            print(f"✅ Health check passed: {data.get('status', 'unknown')}")
            
            # Detailed health check
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get('services', {})
                
                print("📊 Service Status:")
                for service, status in services.items():
                    if isinstance(status, dict):
                        status_str = status.get('status', 'unknown')
                    else:
                        status_str = status
                    print(f"  - {service}: {status_str}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def show_api_info(self):
        """Show API information and endpoints."""
        print("\n" + "="*60)
        print("🎉 Phase 3 API is running!")
        print("="*60)
        
        print(f"🌐 API URL: {self.api_url}")
        print(f"📚 Documentation: {self.api_url}/docs")
        print(f"🔍 Health Check: {self.health_url}")
        
        print("\n📋 Available Endpoints:")
        print("  • GET  /                    - API information")
        print("  • GET  /health              - Health check")
        print("  • GET  /api/v1/health       - Detailed health")
        print("  • POST /api/v1/optimize/workload - Workload optimization")
        print("  • POST /api/v1/predict/workload  - Workload prediction")
        print("  • GET  /api/v1/monitoring/metrics - System metrics")
        
        print("\n🔧 Monitoring Tools:")
        print("  • Grafana: http://localhost:3001")
        print("  • Prometheus: http://localhost:9090")
        print("  • Jaeger: http://localhost:16686")
        
        print("\n🧪 Testing:")
        print("  • python scripts/test_phase3_api.py")
        
        print("\n⏹️  To stop the API: Ctrl+C")
        print("="*60)
    
    def run(self):
        """Run the complete startup process."""
        print("🚀 Phase 3 API Startup")
        print("="*40)
        
        # Check prerequisites
        if not self.check_prerequisites():
            sys.exit(1)
        
        # Setup environment
        self.setup_environment()
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed")
            sys.exit(1)
        
        # Start API
        api_process = self.start_api()
        if not api_process:
            sys.exit(1)
        
        # Wait for API to be ready
        if not self.wait_for_api():
            print("❌ API failed to start properly")
            api_process.terminate()
            sys.exit(1)
        
        # Run health checks
        if not self.run_health_checks():
            print("⚠️  Health checks failed, but API is running")
        
        # Show API information
        self.show_api_info()
        
        try:
            # Keep the process running
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down API...")
            api_process.terminate()
            api_process.wait()
            print("✅ API stopped")


def main():
    """Main function."""
    startup = Phase3Startup()
    startup.run()


if __name__ == "__main__":
    main()
