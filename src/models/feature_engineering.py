"""
Feature Engineering for Healthcare Data Pipeline.

This module provides feature engineering utilities for healthcare data,
including data preprocessing, feature extraction, and transformation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering utilities for healthcare data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def extract_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from input data for prediction.
        
        Args:
            input_data: Dictionary containing healthcare data
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic numerical features
        if 'patient_count' in input_data:
            features['patient_count'] = float(input_data['patient_count'])
        else:
            features['patient_count'] = 0.0
            
        if 'staff_count' in input_data:
            features['staff_count'] = float(input_data['staff_count'])
        else:
            features['staff_count'] = 0.0
            
        if 'bed_count' in input_data:
            features['bed_count'] = float(input_data['bed_count'])
        else:
            features['bed_count'] = 0.0
        
        # Time-based features
        current_time = datetime.now()
        features['hour'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['month'] = current_time.month
        features['is_weekend'] = 1.0 if current_time.weekday() >= 5 else 0.0
        
        # Seasonal features
        features['is_holiday_season'] = 1.0 if current_time.month in [11, 12] else 0.0
        features['is_summer'] = 1.0 if current_time.month in [6, 7, 8] else 0.0
        
        # Derived features
        if features['staff_count'] > 0:
            features['patient_staff_ratio'] = features['patient_count'] / features['staff_count']
        else:
            features['patient_staff_ratio'] = 0.0
            
        if features['bed_count'] > 0:
            features['bed_occupancy_rate'] = features['patient_count'] / features['bed_count']
        else:
            features['bed_occupancy_rate'] = 0.0
        
        # Workload indicators
        features['peak_hour'] = 1.0 if 8 <= features['hour'] <= 18 else 0.0
        features['emergency_hour'] = 1.0 if features['hour'] in [0, 1, 2, 3, 4, 5, 6, 7, 22, 23] else 0.0
        
        # Normalize features to reasonable ranges
        features['patient_count_norm'] = min(features['patient_count'] / 100.0, 1.0)
        features['staff_count_norm'] = min(features['staff_count'] / 50.0, 1.0)
        
        return features
    
    def prepare_training_data(self, data_path: str = "data/processed/parquet/encounters.parquet") -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from healthcare encounters.
        
        Args:
            data_path: Path to encounters data
            
        Returns:
            Tuple of (X, y) for training
        """
        try:
            # Load encounters data
            encounters_df = pd.read_parquet(data_path)
            
            # Sample data for faster processing
            if len(encounters_df) > 10000:
                encounters_df = encounters_df.sample(n=10000, random_state=42)
            
            # Create features
            features_list = []
            targets = []
            
            # Group by hour and create features
            encounters_df['START'] = pd.to_datetime(encounters_df['START'])
            encounters_df['hour'] = encounters_df['START'].dt.hour
            encounters_df['day_of_week'] = encounters_df['START'].dt.dayofweek
            encounters_df['month'] = encounters_df['START'].dt.month
            
            # Aggregate by hour
            hourly_data = encounters_df.groupby([
                encounters_df['START'].dt.date,
                encounters_df['START'].dt.hour
            ]).size().reset_index(name='encounter_count')
            
            hourly_data['date'] = hourly_data['level_0']
            hourly_data['hour'] = hourly_data['level_1']
            
            for _, row in hourly_data.iterrows():
                # Create features for this hour
                features = {
                    'hour': row['hour'],
                    'day_of_week': pd.to_datetime(row['date']).weekday(),
                    'month': pd.to_datetime(row['date']).month,
                    'is_weekend': 1.0 if pd.to_datetime(row['date']).weekday() >= 5 else 0.0,
                    'is_holiday_season': 1.0 if pd.to_datetime(row['date']).month in [11, 12] else 0.0,
                    'is_summer': 1.0 if pd.to_datetime(row['date']).month in [6, 7, 8] else 0.0,
                    'peak_hour': 1.0 if 8 <= row['hour'] <= 18 else 0.0,
                    'emergency_hour': 1.0 if row['hour'] in [0, 1, 2, 3, 4, 5, 6, 7, 22, 23] else 0.0
                }
                
                features_list.append(list(features.values()))
                targets.append(row['encounter_count'])
            
            X = np.array(features_list)
            y = np.array(targets)
            
            self.feature_names = list(features.keys())
            
            self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            # Return synthetic data if real data is not available
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for testing."""
        np.random.seed(42)
        
        n_samples = 1000
        
        # Generate synthetic features
        hours = np.random.randint(0, 24, n_samples)
        days_of_week = np.random.randint(0, 7, n_samples)
        months = np.random.randint(1, 13, n_samples)
        
        # Create features
        X = np.column_stack([
            hours,
            days_of_week,
            months,
            (days_of_week >= 5).astype(float),  # is_weekend
            ((months == 11) | (months == 12)).astype(float),  # is_holiday_season
            ((months >= 6) & (months <= 8)).astype(float),  # is_summer
            ((hours >= 8) & (hours <= 18)).astype(float),  # peak_hour
            ((hours <= 7) | (hours >= 22)).astype(float),  # emergency_hour
        ])
        
        # Generate synthetic targets (encounter counts)
        base_encounters = 50
        hour_factor = 1.5 * np.sin(2 * np.pi * hours / 24) + 1.0
        day_factor = 0.8 + 0.4 * np.sin(2 * np.pi * days_of_week / 7)
        month_factor = 1.0 + 0.3 * np.sin(2 * np.pi * months / 12)
        
        y = base_encounters * hour_factor * day_factor * month_factor + np.random.normal(0, 10, n_samples)
        y = np.maximum(y, 0)  # Ensure non-negative
        
        self.feature_names = ['hour', 'day_of_week', 'month', 'is_weekend', 
                             'is_holiday_season', 'is_summer', 'peak_hour', 'emergency_hour']
        
        self.logger.info(f"Generated synthetic training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
