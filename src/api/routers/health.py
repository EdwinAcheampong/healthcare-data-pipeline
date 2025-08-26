"""
Health check router for the Healthcare Data Pipeline API.

This module provides endpoints for monitoring the health and status
of the API and its dependencies.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import time
import psutil
import redis
import sqlalchemy
from datetime import datetime
from typing import Dict, Any

from src.config.settings import get_settings
from src.api.models.responses import HealthResponse, ResponseStatus
from src.utils.logging import setup_logging

logger = setup_logging()
router = APIRouter()

# Global variables for tracking startup time and service status
startup_time = time.time()
service_status = {
    "database": "unknown",
    "cache": "unknown", 
    "models": "unknown"
}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    settings = get_settings()
    
    try:
        # Check database connection
        db_status = await check_database_health()
        
        # Check cache connection
        cache_status = await check_cache_health()
        
        # Check model availability
        model_status = await check_model_health()
        
        # Calculate uptime
        uptime = time.time() - startup_time
        
        # Determine overall status
        overall_status = "healthy"
        if any(status == "unhealthy" for status in [db_status, cache_status, model_status]):
            overall_status = "degraded"
        if any(status == "error" for status in [db_status, cache_status, model_status]):
            overall_status = "unhealthy"
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Build services health dict
        services = {
            "database": {
                "status": db_status,
                "response_time": system_metrics.get("db_response_time", 0)
            },
            "cache": {
                "status": cache_status,
                "response_time": system_metrics.get("cache_response_time", 0)
            },
            "models": {
                "status": model_status,
                "available_models": system_metrics.get("available_models", [])
            },
            "system": {
                "cpu_usage": system_metrics.get("cpu_usage", 0),
                "memory_usage": system_metrics.get("memory_usage", 0),
                "disk_usage": system_metrics.get("disk_usage", 0)
            }
        }
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="3.0.0",
            environment=settings.environment,
            uptime=uptime,
            services=services,
            database_status=db_status,
            cache_status=cache_status,
            model_status=model_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/simple")
async def simple_health_check():
    """Simple health check endpoint for load balancers."""
    try:
        # Quick checks
        db_ok = await check_database_health() == "healthy"
        cache_ok = await check_cache_health() == "healthy"
        
        if db_ok and cache_ok:
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy"}
            
    except Exception:
        return {"status": "unhealthy"}


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes deployments."""
    try:
        # Check if all critical services are ready
        db_ready = await check_database_health() == "healthy"
        cache_ready = await check_cache_health() == "healthy"
        models_ready = await check_model_health() in ["healthy", "degraded"]
        
        if db_ready and cache_ready and models_ready:
            return {"ready": True}
        else:
            return {"ready": False, "reason": "Services not ready"}
            
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes deployments."""
    try:
        # Simple check if the application is running
        return {"alive": True, "timestamp": datetime.now()}
    except Exception:
        return {"alive": False}


@router.get("/status")
async def system_status():
    """Detailed system status endpoint."""
    try:
        settings = get_settings()
        
        # Get comprehensive system metrics
        metrics = get_system_metrics()
        
        # Get service status
        services = {
            "database": await check_database_health(),
            "cache": await check_cache_health(),
            "models": await check_model_health(),
            "api": "healthy"
        }
        
        # Calculate performance metrics
        performance = {
            "response_time_avg": metrics.get("avg_response_time", 0),
            "requests_per_second": metrics.get("requests_per_second", 0),
            "error_rate": metrics.get("error_rate", 0),
            "uptime": time.time() - startup_time
        }
        
        return {
            "status": "operational",
            "timestamp": datetime.now(),
            "version": "3.0.0",
            "environment": settings.environment,
            "services": services,
            "performance": performance,
            "system": {
                "cpu_usage": metrics.get("cpu_usage", 0),
                "memory_usage": metrics.get("memory_usage", 0),
                "disk_usage": metrics.get("disk_usage", 0),
                "network_io": metrics.get("network_io", {})
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Status check failed")


async def check_database_health() -> str:
    """Check database connection health."""
    try:
        settings = get_settings()
        
        # Try to connect to database
        if hasattr(settings, 'database_url') and settings.database_url:
            engine = sqlalchemy.create_engine(settings.database_url)
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            return "healthy"
        else:
            return "not_configured"
            
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return "unhealthy"


async def check_cache_health() -> str:
    """Check cache connection health."""
    try:
        settings = get_settings()
        
        # Try to connect to Redis
        if hasattr(settings, 'redis_url') and settings.redis_url:
            r = redis.from_url(settings.redis_url)
            r.ping()
            return "healthy"
        else:
            return "not_configured"
            
    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
        return "unhealthy"


async def check_model_health() -> str:
    """Check ML model availability."""
    try:
        # Check if model files exist and are accessible
        import os
        from pathlib import Path
        
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            if model_files:
                return "healthy"
            else:
                return "no_models"
        else:
            return "not_configured"
            
    except Exception as e:
        logger.error(f"Model health check failed: {str(e)}")
        return "error"


def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_io": network_io,
            "available_models": ["baseline", "advanced", "rl_optimization"],  # Placeholder
            "avg_response_time": 0.15,  # Placeholder
            "requests_per_second": 10.5,  # Placeholder
            "error_rate": 0.001  # Placeholder
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        return {}


@router.get("/info")
async def api_info():
    """Get API information and capabilities."""
    return {
        "name": "Healthcare Data Pipeline API",
        "version": "3.0.0",
        "description": "Production API for healthcare workload optimization and ML predictions",
        "endpoints": {
            "optimization": "/api/v1/optimize/*",
            "predictions": "/api/v1/predict/*",
            "monitoring": "/api/v1/monitoring/*",
            "health": "/api/v1/health/*"
        },
        "features": [
            "Real-time workload optimization",
            "ML-powered predictions",
            "Comprehensive monitoring",
            "Health checks and alerts"
        ],
        "documentation": "/docs",
        "status": "operational"
    }
