"""
Request models for the Healthcare Data Pipeline API.

This module contains Pydantic models for validating incoming requests
to the API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class ModelType(str, Enum):
    """Available model types for predictions."""
    BASELINE = "baseline"
    ADVANCED = "advanced"
    RL_OPTIMIZATION = "rl_optimization"


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    CUSTOM = "custom"


class WorkloadOptimizationRequest(BaseModel):
    """Request model for workload optimization."""
    
    current_patients: int = Field(..., ge=0, le=1000, description="Current number of patients")
    current_staff: int = Field(..., ge=0, le=500, description="Current number of staff")
    department: str = Field(..., min_length=1, max_length=100, description="Department name")
    shift_hours: int = Field(8, ge=1, le=24, description="Shift duration in hours")
    optimization_strategy: OptimizationStrategy = Field(
        OptimizationStrategy.PPO, 
        description="Optimization strategy to use"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional constraints for optimization"
    )
    target_metrics: Optional[List[str]] = Field(
        default_factory=lambda: ["efficiency", "patient_satisfaction", "staff_workload"],
        description="Target metrics to optimize"
    )
    
    @validator('current_patients')
    def validate_patients(cls, v):
        if v < 0:
            raise ValueError('Patient count must be non-negative')
        return v
    
    @validator('current_staff')
    def validate_staff(cls, v):
        if v < 0:
            raise ValueError('Staff count must be non-negative')
        return v


class PredictionRequest(BaseModel):
    """Request model for ML predictions."""
    
    model_type: ModelType = Field(..., description="Type of model to use for prediction")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    features: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific features to use for prediction"
    )
    prediction_horizon: Optional[int] = Field(
        24, 
        ge=1, 
        le=168, 
        description="Prediction horizon in hours (1-168)"
    )
    confidence_level: Optional[float] = Field(
        0.95, 
        ge=0.5, 
        le=0.99, 
        description="Confidence level for prediction intervals"
    )
    
    @validator('input_data')
    def validate_input_data(cls, v):
        if not v:
            raise ValueError('Input data cannot be empty')
        return v


class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    
    model_type: ModelType = Field(..., description="Type of model to train")
    training_data_path: str = Field(..., description="Path to training data")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model hyperparameters"
    )
    training_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Training configuration"
    )
    
    @validator('validation_split')
    def validate_validation_split(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('Validation split must be between 0 and 1')
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    model_type: ModelType = Field(..., description="Type of model to use")
    input_data_list: List[Dict[str, Any]] = Field(..., description="List of input data")
    batch_size: Optional[int] = Field(100, ge=1, le=1000, description="Batch size for processing")
    
    @validator('input_data_list')
    def validate_input_data_list(cls, v):
        if not v:
            raise ValueError('Input data list cannot be empty')
        if len(v) > 10000:
            raise ValueError('Too many predictions requested (max 10000)')
        return v


class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    
    model_type: ModelType = Field(..., description="Type of model to evaluate")
    test_data_path: str = Field(..., description="Path to test data")
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ["mape", "rmse", "mae"],
        description="Metrics to calculate"
    )
    comparison_models: Optional[List[str]] = Field(
        default_factory=list,
        description="Other models to compare against"
    )


class DataValidationRequest(BaseModel):
    """Request model for data validation."""
    
    data_path: str = Field(..., description="Path to data for validation")
    validation_rules: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom validation rules"
    )
    output_format: str = Field("json", description="Output format for validation results")
    
    @validator('output_format')
    def validate_output_format(cls, v):
        allowed_formats = ["json", "csv", "html"]
        if v not in allowed_formats:
            raise ValueError(f'Output format must be one of: {allowed_formats}')
        return v
