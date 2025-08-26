"""
Response models for the Healthcare Data Pipeline API.

This module contains Pydantic models for structuring API responses
and ensuring consistent response formats.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    FAILED = "failed"


class OptimizationResponse(BaseModel):
    """Response model for workload optimization."""
    
    optimization_id: str = Field(..., description="Unique optimization job ID")
    status: ResponseStatus = Field(..., description="Optimization status")
    recommended_staff: int = Field(..., description="Recommended staff count")
    recommended_schedule: Dict[str, Any] = Field(..., description="Recommended schedule")
    efficiency_gain: float = Field(..., description="Expected efficiency improvement (%)")
    patient_satisfaction_improvement: float = Field(..., description="Expected patient satisfaction improvement (%)")
    staff_workload_reduction: float = Field(..., description="Expected staff workload reduction (%)")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class PredictionResponse(BaseModel):
    """Response model for ML predictions."""
    
    prediction_id: str = Field(..., description="Unique prediction ID")
    model_type: str = Field(..., description="Model type used")
    predicted_value: Union[float, int, str] = Field(..., description="Predicted value")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")
    prediction_horizon: int = Field(..., description="Prediction horizon in hours")
    features_used: List[str] = Field(..., description="Features used for prediction")
    model_confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    batch_id: str = Field(..., description="Unique batch ID")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of predictions processed")
    success_count: int = Field(..., description="Number of successful predictions")
    error_count: int = Field(..., description="Number of failed predictions")
    processing_time: float = Field(..., description="Total processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ModelPerformanceResponse(BaseModel):
    """Response model for model performance metrics."""
    
    model_type: str = Field(..., description="Model type")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    training_time: float = Field(..., description="Training time in seconds")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")
    model_size: Optional[float] = Field(None, description="Model size in MB")
    version: str = Field(..., description="Model version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    comparison_results: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="Comparison with other models"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    uptime: float = Field(..., description="Service uptime in seconds")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service health")
    database_status: str = Field(..., description="Database connection status")
    cache_status: str = Field(..., description="Cache connection status")
    model_status: str = Field(..., description="ML model status")


class DataValidationResponse(BaseModel):
    """Response model for data validation."""
    
    validation_id: str = Field(..., description="Unique validation ID")
    status: ResponseStatus = Field(..., description="Validation status")
    total_records: int = Field(..., description="Total records validated")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    validation_errors: List[Dict[str, Any]] = Field(..., description="Validation errors found")
    data_quality_score: float = Field(..., ge=0, le=1, description="Overall data quality score")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ModelTrainingResponse(BaseModel):
    """Response model for model training."""
    
    training_id: str = Field(..., description="Unique training job ID")
    status: ResponseStatus = Field(..., description="Training status")
    model_type: str = Field(..., description="Model type being trained")
    training_progress: float = Field(..., ge=0, le=1, description="Training progress (0-1)")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total training epochs")
    current_metrics: Optional[Dict[str, float]] = Field(None, description="Current training metrics")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class StatusResponse(BaseModel):
    """Response model for job status checks."""
    
    job_id: str = Field(..., description="Job ID")
    status: ResponseStatus = Field(..., description="Job status")
    progress: float = Field(..., ge=0, le=1, description="Job progress (0-1)")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    api_metrics: Dict[str, Any] = Field(..., description="API performance metrics")
    model_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    system_metrics: Dict[str, Any] = Field(..., description="System resource metrics")
    business_metrics: Dict[str, Any] = Field(..., description="Business KPIs")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active alerts")
