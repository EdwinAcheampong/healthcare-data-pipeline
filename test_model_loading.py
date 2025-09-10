#!/usr/bin/env python3
"""
Test script to check if models can be loaded properly.
"""

import sys
from pathlib import Path
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    # Import the model classes
    from src.models.baseline_models import BaselinePredictor
    from src.models.advanced_models import AdvancedPredictor
    print("SUCCESS: Model classes imported successfully")
    
    # Try to load the saved models
    models_path = project_root / "models"
    
    with open(models_path / "baseline_predictor.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    print("SUCCESS: Baseline model loaded successfully")
    
    with open(models_path / "advanced_predictor.pkl", "rb") as f:
        advanced_model = pickle.load(f)
    print("SUCCESS: Advanced model loaded successfully")
    
    with open(models_path / "baseline_scaler.pkl", "rb") as f:
        baseline_scaler = pickle.load(f)
    print("SUCCESS: Baseline scaler loaded successfully")
    
    with open(models_path / "advanced_scaler.pkl", "rb") as f:
        advanced_scaler = pickle.load(f)
    print("SUCCESS: Advanced scaler loaded successfully")
    
    print("\nSUCCESS: All models loaded successfully! The import issue is fixed.")
    
except ImportError as e:
    print(f"ERROR: Import error: {e}")
except FileNotFoundError as e:
    print(f"ERROR: File not found: {e}")
except Exception as e:
    print(f"ERROR: Error loading models: {e}")