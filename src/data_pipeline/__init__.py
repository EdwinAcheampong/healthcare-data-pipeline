"""
Healthcare Data Pipeline Package.

This package provides a comprehensive data processing pipeline for healthcare data,
including ETL, feature engineering, validation, and storage management.
"""

from .etl import ETLPipeline, DataQualityMetrics
from .storage import (
    StorageManager,
    ParquetStorage,
    FeatureStore
)

__version__ = "0.1.0"
__author__ = "Muhammad Yekini"

__all__ = [
    'ETLPipeline',
    'DataQualityMetrics',
    'StorageManager',
    'ParquetStorage',
    'FeatureStore'
]