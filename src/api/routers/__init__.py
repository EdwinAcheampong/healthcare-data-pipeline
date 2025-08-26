"""
API routers package for Phase 3.

This package contains all the FastAPI routers for different API endpoints.
"""

from .health import router as health_router
from .optimization import router as optimization_router
from .predictions import router as predictions_router
from .monitoring import router as monitoring_router

__all__ = [
    "health_router",
    "optimization_router", 
    "predictions_router",
    "monitoring_router"
]
