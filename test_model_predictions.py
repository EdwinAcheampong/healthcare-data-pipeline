#!/usr/bin/env python3
"""
Test script to manually check if models can be unpickled and make predictions.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_model_predictions():
    """Test loading models and making predictions."""
    try:
        # Import the model classes
        from src.models.baseline_models import BaselinePredictor
        from src.models.advanced_models import AdvancedPredictor
        print("SUCCESS: Model classes imported successfully")
        
        # Load the saved models
        models_path = project_root / "models"
        
        # Load baseline model and scaler
        with open(models_path / "baseline_predictor.pkl", "rb") as f:
            baseline_model = pickle.load(f)
        print("SUCCESS: Baseline model loaded successfully")
        
        with open(models_path / "baseline_scaler.pkl", "rb") as f:
            baseline_scaler = pickle.load(f)
        print("SUCCESS: Baseline scaler loaded successfully")
        
        # Load advanced model and scaler
        with open(models_path / "advanced_predictor.pkl", "rb") as f:
            advanced_model = pickle.load(f)
        print("SUCCESS: Advanced model loaded successfully")
        
        with open(models_path / "advanced_scaler.pkl", "rb") as f:
            advanced_scaler = pickle.load(f)
        print("SUCCESS: Advanced scaler loaded successfully")
        
        # Test predictions with sample data
        print("\n--- Testing Model Predictions ---")
        
        # Create sample input data (6 features as per the ML script)
        # Features: ['age', 'encounter_count', 'condition_count', 'medication_count', 'avg_duration', 'healthcare_expenses']
        sample_data = np.array([[
            45.0,    # age
            5.0,     # encounter_count
            2.0,     # condition_count
            3.0,     # medication_count
            2.5,     # avg_duration
            1500.0   # healthcare_expenses
        ]])
        
        print(f"Sample input data shape: {sample_data.shape}")
        print(f"Sample input data: {sample_data[0]}")
        
        # Scale the data
        scaled_data = baseline_scaler.transform(sample_data)
        print(f"Scaled data: {scaled_data[0]}")
        
        # Make predictions
        baseline_prediction = baseline_model.predict(scaled_data)
        print(f"Baseline model prediction: {baseline_prediction[0]:.2f}")
        
        # Test advanced model with its scaler
        advanced_scaled_data = advanced_scaler.transform(sample_data)
        advanced_prediction = advanced_model.predict(advanced_scaled_data)
        print(f"Advanced model prediction: {advanced_prediction[0]:.2f}")
        
        # Test feature importance
        if hasattr(baseline_model, 'get_feature_importance'):
            feature_importance = baseline_model.get_feature_importance()
            print(f"Feature importance: {feature_importance}")
        
        print("\nSUCCESS: All model predictions working correctly!")
        print("The models are properly saved and can be used for predictions.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_predictions()
    if success:
        print("\nCONCLUSION: Models are working correctly. The dashboard should be able to load them.")
    else:
        print("\nCONCLUSION: There are issues with the models that need to be fixed.")