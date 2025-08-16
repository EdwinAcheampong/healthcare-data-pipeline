"""
Healthcare Data Pipeline Package.

This package provides a comprehensive data processing pipeline for healthcare data,
including ETL, feature engineering, validation, and storage management.
"""

from .etl import ETLPipeline, DataQualityMetrics
from .feature_engineering import (
    ClinicalFeatureEngineer,
    InfrastructureFeatureEngineer, 
    TemporalFeatureEngineer,
    FeatureStore,
    FeatureEngineeringPipeline
)
from .validation import (
    DataValidationPipeline,
    ValidationResult,
    ValidationLevel,
    DataQualityReport
)
from .storage import (
    StorageManager,
    ParquetStorage,
    TimeSeriesStorage,
    FeatureStore as FeatureStoreStorage
)

__version__ = "0.1.0"
__author__ = "Muhammad Yekini"

__all__ = [
    'ETLPipeline',
    'DataQualityMetrics',
    'ClinicalFeatureEngineer',
    'InfrastructureFeatureEngineer',
    'TemporalFeatureEngineer',
    'FeatureStore',
    'FeatureEngineeringPipeline',
    'DataValidationPipeline',
    'ValidationResult',
    'ValidationLevel',
    'DataQualityReport',
    'StorageManager',
    'ParquetStorage',
    'TimeSeriesStorage',
    'FeatureStoreStorage'
]