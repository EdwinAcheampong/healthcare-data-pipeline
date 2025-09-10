#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Test model imports
        from src.models.baseline_models import BaselinePredictor
        print("✓ BaselinePredictor imported successfully")
        
        from src.models.advanced_models import AdvancedPredictor
        print("✓ AdvancedPredictor imported successfully")
        
        from src.models.feature_engineering import FeatureEngineer
        print("✓ FeatureEngineer imported successfully")
        
        from src.models.rl_environment import HealthcareWorkloadEnvironment
        print("✓ HealthcareWorkloadEnvironment imported successfully")
        
        from src.models.ppo_agent import PPOHHealthcareAgent, PPOConfig
        print("✓ PPO Agent imported successfully")
        
        # Test config imports
        try:
            from src.config.settings import settings
            print("✓ Settings imported successfully")
        except ImportError as e:
            print(f"⚠ Settings import failed (this might be expected): {e}")
        
        # Test data pipeline imports
        from src.data_pipeline.storage import StorageManager
        print("✓ StorageManager imported successfully")
        
        print("\n🎉 All critical imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)