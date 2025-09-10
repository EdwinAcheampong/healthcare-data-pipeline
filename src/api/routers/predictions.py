"""
Predictions router for the Healthcare Data Pipeline API.

This module provides endpoints for ML predictions using
the models developed in Phase 2A.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from src.config.settings import get_settings
from src.api.models.requests import (
    PredictionRequest, 
    BatchPredictionRequest,
    ModelTrainingRequest,
    ModelEvaluationRequest
)
from src.api.models.responses import (
    PredictionResponse,
    BatchPredictionResponse,
    ModelPerformanceResponse,
    ModelTrainingResponse,
    StatusResponse,
    ResponseStatus
)
from src.api.models.schemas import PredictionInput, ModelMetadata
from src.utils.logging import setup_logging
from src.models.baseline_models import BaselinePredictor
from src.models.advanced_models import AdvancedPredictor
from src.models.feature_engineering import FeatureEngineer

# Setup logger first
logger = setup_logging()

# Enable ML models
ML_AVAILABLE = True
logger.info("ML models enabled.")

# Load models and scalers at startup
models_path = Path(__file__).resolve().parent.parent.parent.parent / "models"
try:
    with open(models_path / "baseline_predictor.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    with open(models_path / "baseline_scaler.pkl", "rb") as f:
        baseline_scaler = pickle.load(f)
    with open(models_path / "advanced_predictor.pkl", "rb") as f:
        advanced_model = pickle.load(f)
    with open(models_path / "advanced_scaler.pkl", "rb") as f:
        advanced_scaler = pickle.load(f)
    logger.info("Successfully loaded ML models and scalers.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    ML_AVAILABLE = False

router = APIRouter()

# In-memory storage for prediction jobs (in production, use Redis/Database)
prediction_jobs = {}
training_jobs = {}


@router.post("/workload", response_model=PredictionResponse)
async def predict_workload(request: PredictionRequest):
    """
    Predict healthcare workload using ML models.
    
    This endpoint uses the ML models from Phase 2A to predict
    various healthcare metrics like patient volume, wait times, etc.
    """
    start_time = time.time()
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML models are not available.")

    try:
        # Validate input data
        if not request.input_data:
            raise HTTPException(
                status_code=400,
                detail="Input data is required"
            )
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Select model based on request
        if request.model_type.value == "baseline":
            model = baseline_model
            scaler = baseline_scaler
        elif request.model_type.value == "advanced":
            model = advanced_model
            scaler = advanced_scaler
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {request.model_type}"
            )
        
        # Prepare features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.extract_features(request.input_data)
        feature_values = np.array(list(features.values())).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(feature_values)
        
        # Make prediction
        prediction_result = model.predict(scaled_features)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction_id=prediction_id,
            model_type=request.model_type.value,
            predicted_value=prediction_result[0],
            confidence_interval=None,  # Replace with actual confidence interval if available
            prediction_horizon=request.prediction_horizon or 24,
            features_used=list(features.keys()),
            model_confidence=0.9,  # Replace with actual confidence if available
            processing_time=processing_time,
            created_at=datetime.now(),
            metadata={
                "model_version": "1.0",
                "feature_importance": {} # Replace with actual feature importance if available
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))