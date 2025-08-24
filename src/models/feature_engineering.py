"""
Feature Engineering for Healthcare Data Analytics.

This module implements comprehensive feature engineering for healthcare workload prediction,
including temporal features, patient demographics, clinical indicators, and resource utilization metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class HealthcareFeatureEngineer:
    """Feature engineering for healthcare workload prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame, 
                               date_column: str = 'START') -> pd.DataFrame:
        """Create temporal features from datetime columns."""
        self.logger.info("Creating temporal features")
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Basic temporal features
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['year'] = df[date_column].dt.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Business hours indicator (8 AM - 6 PM)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Emergency hours indicator (6 PM - 8 AM)
        df['is_emergency_hours'] = ((df['hour'] < 8) | (df['hour'] >= 18)).astype(int)
        
        return df
    
    def create_patient_features(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Create patient demographic and clinical features."""
        self.logger.info("Creating patient features")
        
        df = patients_df.copy()
        
        # Age features
        if 'BIRTHDATE' in df.columns:
            df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
            df['age'] = (datetime.now() - df['BIRTHDATE']).dt.days / 365.25
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 18, 30, 50, 65, 100], 
                                   labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])
        
        # Gender encoding
        if 'GENDER' in df.columns:
            df['gender_encoded'] = (df['GENDER'] == 'M').astype(int)
        
        # Race/Ethnicity encoding
        if 'RACE' in df.columns:
            le = LabelEncoder()
            df['race_encoded'] = le.fit_transform(df['RACE'].fillna('Unknown'))
            self.label_encoders['race'] = le
        
        # Insurance status
        if 'PASSPORT' in df.columns:
            df['has_insurance'] = df['PASSPORT'].notna().astype(int)
        
        return df
    
    def create_encounter_features(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """Create encounter-based features."""
        self.logger.info("Creating encounter features")
        
        df = encounters_df.copy()
        
        # Encounter duration
        if 'START' in df.columns and 'STOP' in df.columns:
            df['START'] = pd.to_datetime(df['START'])
            df['STOP'] = pd.to_datetime(df['STOP'])
            df['encounter_duration_hours'] = (df['STOP'] - df['START']).dt.total_seconds() / 3600
            df['encounter_duration_hours'] = df['encounter_duration_hours'].fillna(0)
        
        # Encounter type encoding
        if 'ENCOUNTERCLASS' in df.columns:
            le = LabelEncoder()
            df['encounter_class_encoded'] = le.fit_transform(df['ENCOUNTERCLASS'].fillna('unknown'))
            self.label_encoders['encounter_class'] = le
        
        # Provider features
        if 'PROVIDER' in df.columns:
            df['provider_id_encoded'] = df['PROVIDER'].astype('category').cat.codes
        
        return df
    
    def create_condition_features(self, conditions_df: pd.DataFrame) -> pd.DataFrame:
        """Create condition-based features."""
        self.logger.info("Creating condition features")
        
        df = conditions_df.copy()
        
        # COVID-19 indicators
        covid_keywords = ['covid', 'coronavirus', 'sars-cov-2', 'covid-19']
        df['is_covid_related'] = df['DESCRIPTION'].str.lower().str.contains(
            '|'.join(covid_keywords), na=False
        ).astype(int)
        
        # Chronic condition indicators
        chronic_keywords = ['diabetes', 'hypertension', 'heart disease', 'asthma', 'copd']
        df['is_chronic_condition'] = df['DESCRIPTION'].str.lower().str.contains(
            '|'.join(chronic_keywords), na=False
        ).astype(int)
        
        # Emergency condition indicators
        emergency_keywords = ['emergency', 'acute', 'severe', 'critical']
        df['is_emergency_condition'] = df['DESCRIPTION'].str.lower().str.contains(
            '|'.join(emergency_keywords), na=False
        ).astype(int)
        
        # Condition severity (based on description length and keywords)
        df['condition_severity'] = df['DESCRIPTION'].str.len() / 100  # Normalize
        
        return df
    
    def create_medication_features(self, medications_df: pd.DataFrame) -> pd.DataFrame:
        """Create medication-based features."""
        self.logger.info("Creating medication features")
        
        df = medications_df.copy()
        
        # Medication count per encounter
        encounter_col = 'ENCOUNTER' if 'ENCOUNTER' in df.columns else 'Id'
        if encounter_col in df.columns:
            med_count = df.groupby(encounter_col).size().reset_index(name='medication_count')
            df = df.merge(med_count, on=encounter_col, how='left')
        
        # Antibiotic indicators
        antibiotic_keywords = ['antibiotic', 'penicillin', 'amoxicillin', 'azithromycin']
        df['is_antibiotic'] = df['DESCRIPTION'].str.lower().str.contains(
            '|'.join(antibiotic_keywords), na=False
        ).astype(int)
        
        # Pain medication indicators
        pain_keywords = ['pain', 'analgesic', 'morphine', 'acetaminophen', 'ibuprofen']
        df['is_pain_medication'] = df['DESCRIPTION'].str.lower().str.contains(
            '|'.join(pain_keywords), na=False
        ).astype(int)
        
        return df
    
    def create_workload_features(self, encounters_df: pd.DataFrame, 
                               window_hours: int = 24) -> pd.DataFrame:
        """Create workload prediction features."""
        self.logger.info("Creating workload features")
        
        df = encounters_df.copy()
        
        # Handle timezone-aware datetime by converting to naive datetime
        if df['START'].dt.tz is not None:
            df['START'] = df['START'].dt.tz_localize(None)
        
        # Sort by time
        df = df.sort_values('START')
        
        # Simplified workload features - avoid complex rolling operations
        encounter_id_col = 'Id' if 'Id' in df.columns else 'ENCOUNTER'
        
        # Calculate daily encounter counts as a simpler alternative
        df['date'] = df['START'].dt.date
        daily_counts = df.groupby(['ORGANIZATION', 'date']).size().reset_index(name='daily_encounters')
        
        # Merge daily counts back to encounters
        df = df.merge(daily_counts, on=['ORGANIZATION', 'date'], how='left')
        df['encounters_last_24h'] = df['daily_encounters'].fillna(0)
        
        # Remove temporary date column
        df = df.drop('date', axis=1)
        
        # Average encounters per hour (simplified calculation)
        df['avg_encounters_per_hour'] = df['encounters_last_24h'] / 24
        
        # Peak hours indicator
        df['is_peak_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def create_aggregated_features(self, encounters_df: pd.DataFrame,
                                 conditions_df: pd.DataFrame,
                                 medications_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features at encounter level."""
        self.logger.info("Creating aggregated features")
        
        # Determine encounter column names
        encounters_encounter_col = 'Id'  # Encounters dataframe uses 'Id'
        conditions_encounter_col = 'ENCOUNTER' if 'ENCOUNTER' in conditions_df.columns else 'Id'
        medications_encounter_col = 'ENCOUNTER' if 'ENCOUNTER' in medications_df.columns else 'Id'
        
        # Aggregate conditions per encounter
        condition_agg = conditions_df.groupby(conditions_encounter_col).agg({
            'is_covid_related': 'sum',
            'is_chronic_condition': 'sum',
            'is_emergency_condition': 'sum',
            'condition_severity': 'mean'
        }).reset_index()
        
        # Rename the encounter column to match encounters dataframe
        condition_agg = condition_agg.rename(columns={conditions_encounter_col: encounters_encounter_col})
        
        # Aggregate medications per encounter
        med_agg = medications_df.groupby(medications_encounter_col).agg({
            'medication_count': 'first',
            'is_antibiotic': 'sum',
            'is_pain_medication': 'sum'
        }).reset_index()
        
        # Rename the encounter column to match encounters dataframe
        med_agg = med_agg.rename(columns={medications_encounter_col: encounters_encounter_col})
        
        # Merge with encounters
        df = encounters_df.merge(condition_agg, on=encounters_encounter_col, how='left')
        df = df.merge(med_agg, on=encounters_encounter_col, how='left')
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def engineer_features(self, 
                         encounters_df: pd.DataFrame,
                         patients_df: pd.DataFrame,
                         conditions_df: pd.DataFrame,
                         medications_df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        self.logger.info("Starting complete feature engineering pipeline")
        
        # Create individual feature sets
        encounters_with_temporal = self.create_temporal_features(encounters_df)
        encounters_with_encounter_features = self.create_encounter_features(encounters_with_temporal)
        encounters_with_workload = self.create_workload_features(encounters_with_encounter_features)
        
        conditions_with_features = self.create_condition_features(conditions_df)
        medications_with_features = self.create_medication_features(medications_df)
        
        # Create aggregated features
        final_features = self.create_aggregated_features(
            encounters_with_workload,
            conditions_with_features,
            medications_with_features
        )
        
        # Merge with patient features
        patients_with_features = self.create_patient_features(patients_df)
        final_features = final_features.merge(
            patients_with_features[['Id', 'age', 'gender_encoded', 'race_encoded', 'has_insurance']],
            left_on='PATIENT',
            right_on='Id',
            how='left'
        )
        
        # Select final feature columns
        feature_columns = [
            # Temporal features
            'hour', 'day_of_week', 'month', 'year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_business_hours', 'is_emergency_hours', 'is_peak_hours',
            
            # Encounter features
            'encounter_duration_hours', 'encounter_class_encoded', 'provider_id_encoded',
            
            # Workload features
            'encounters_last_24h', 'avg_encounters_per_hour',
            
            # Condition features
            'is_covid_related', 'is_chronic_condition', 'is_emergency_condition', 'condition_severity',
            
            # Medication features
            'medication_count', 'is_antibiotic', 'is_pain_medication',
            
            # Patient features
            'age', 'gender_encoded', 'race_encoded', 'has_insurance'
        ]
        
        # Filter to available columns
        available_columns = [col for col in feature_columns if col in final_features.columns]
        
        # Select required columns that exist
        required_columns = ['Id', 'PATIENT', 'START']
        existing_required_columns = [col for col in required_columns if col in final_features.columns]
        
        final_features = final_features[available_columns + existing_required_columns]
        
        self.feature_names = available_columns
        self.logger.info(f"Feature engineering completed. Created {len(available_columns)} features")
        
        return final_features
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        self.logger.info("Scaling features")
        
        # Select numerical columns (exclude datetime and categorical)
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        encounter_col = 'Id' if 'Id' in df.columns else 'ENCOUNTER'
        numerical_columns = [col for col in numerical_columns if col not in [encounter_col, 'PATIENT']]
        
        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """Select top k features using statistical tests."""
        self.logger.info(f"Selecting top {k} features")
        
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        self.logger.info(f"Selected features: {selected_features}")
        
        return X[selected_features]
