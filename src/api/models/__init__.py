"""
API models package for Phase 3.

This package contains all the Pydantic models used for request/response
validation in the FastAPI application.
"""

from .requests import *
from .responses import *
from .schemas import *

__all__ = [
    # Request models
    "WorkloadOptimizationRequest",
    "PredictionRequest",
    "ModelTrainingRequest",
    
    # Response models
    "OptimizationResponse",
    "PredictionResponse",
    "ModelPerformanceResponse",
    "HealthResponse",
    
    # Schema models
    "PatientData",
    "StaffData",
    "WorkloadMetrics",
    "OptimizationResult"
]
