"""
ETL Pipeline for Healthcare Data Processing.

This module implements the core ETL pipeline for processing Synthea healthcare data
with PHI removal, data validation, and feature engineering capabilities.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import hashlib

from pydantic import BaseModel, validator
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings


class DataQualityMetrics(BaseModel):
    """Data quality metrics for validation."""

    total_records: int
    null_percentage: float
    duplicate_percentage: float
    data_types_valid: bool
    date_range_valid: bool
    phi_removed: bool
    processing_timestamp: datetime

    @validator('null_percentage', 'duplicate_percentage')
    def percentage_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v


class ETLPipeline:
    """Main ETL Pipeline for healthcare data processing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.synthea_path = settings.data.synthea_data_path
        self.processed_path = settings.data.processed_data_path

        # Ensure output directories exist
        self.processed_path.mkdir(parents=True, exist_ok=True)
        (self.processed_path / "parquet").mkdir(exist_ok=True)

        # PHI patterns for removal
        self.phi_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.'
                      r'[A-Z|a-z]{2,}\b'),
            'drivers_license': r'\b[A-Z]\d{8}\b',
            'passport': r'\b[A-Z]\d{8}[A-Z]\b'
        }

    def extract_synthea_data(self) -> Dict[str, pd.DataFrame]:
        """Extract all Synthea CSV files into DataFrames."""
        self.logger.info("Starting data extraction from Synthea files")

        dataframes = {}
        csv_files = list(self.synthea_path.glob("*.csv"))

        for csv_file in csv_files:
            try:
                table_name = csv_file.stem
                self.logger.info(f"Loading {table_name} from {csv_file}")

                df = pd.read_csv(csv_file)
                dataframes[table_name] = df

                self.logger.info(f"Loaded {len(df)} records from {table_name}")

            except Exception as e:
                self.logger.error(f"Error loading {csv_file}: {str(e)}")
                raise

        return dataframes

    def remove_phi(self, df: pd.DataFrame, 
                   columns_to_anonymize: List[str]) -> pd.DataFrame:
        """Remove PHI from specified columns."""
        self.logger.info("Starting PHI removal process")

        df_clean = df.copy()

        for column in columns_to_anonymize:
            if column in df_clean.columns:
                # Hash sensitive identifiers
                if column.lower() in ['ssn', 'drivers', 'passport']:
                    df_clean[column] = df_clean[column].apply(
                        lambda x: self._hash_identifier(str(x)) 
                        if pd.notna(x) else x
                    )

                # Remove text patterns that might contain PHI
                elif df_clean[column].dtype == 'object':
                    for pattern_name, pattern in self.phi_patterns.items():
                        df_clean[column] = df_clean[column].astype(str).str.replace(
                            pattern, f'[{pattern_name.upper()}_REMOVED]', 
                            regex=True
                        )

        self.logger.info(f"PHI removal completed for {len(columns_to_anonymize)} columns")
        return df_clean

    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier for anonymization."""
        if not identifier or identifier == 'nan':
            return identifier
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def validate_data_quality(self, df: pd.DataFrame, 
                             table_name: str) -> DataQualityMetrics:
        """Validate data quality and return metrics."""
        self.logger.info(f"Validating data quality for {table_name}")

        total_records = len(df)
        null_count = df.isnull().sum().sum()
        total_cells = df.size
        null_percentage = (null_count / total_cells) * 100 if total_cells > 0 else 0

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / total_records) * 100 if total_records > 0 else 0

        # Validate data types
        data_types_valid = self._validate_data_types(df, table_name)

        # Validate date ranges
        date_range_valid = self._validate_date_ranges(df)

        # Check PHI removal (simplified check)
        phi_removed = self._check_phi_removal(df)

        metrics = DataQualityMetrics(
            total_records=total_records,
            null_percentage=null_percentage,
            duplicate_percentage=duplicate_percentage,
            data_types_valid=data_types_valid,
            date_range_valid=date_range_valid,
            phi_removed=phi_removed,
            processing_timestamp=datetime.now()
        )

        self.logger.info(f"Data quality validation completed for {table_name}")
        return metrics

    def _validate_data_types(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate that data types are appropriate."""
        try:
            # Check for date columns
            date_columns = [col for col in df.columns if any(
                date_word in col.lower() 
                for date_word in ['date', 'start', 'stop', 'birth']
            )]

            for col in date_columns:
                if col in df.columns:
                    # Try to parse dates
                    pd.to_datetime(df[col], errors='coerce')

            return True
        except Exception as e:
            self.logger.warning(f"Data type validation failed for {table_name}: {str(e)}")
            return False

    def _validate_date_ranges(self, df: pd.DataFrame) -> bool:
        """Validate that dates are within reasonable ranges."""
        try:
            date_columns = [col for col in df.columns if any(
                date_word in col.lower() 
                for date_word in ['date', 'start', 'stop', 'birth']
            )]

            current_year = datetime.now().year
            min_year = 1900

            for col in date_columns:
                if col in df.columns:
                    dates = pd.to_datetime(df[col], errors='coerce').dropna()
                    if not dates.empty:
                        years = dates.dt.year
                        if (years < min_year).any() or (years > current_year + 1).any():
                            return False

            return True
        except Exception as e:
            self.logger.warning(f"Date range validation failed: {str(e)}")
            return False

    def _check_phi_removal(self, df: pd.DataFrame) -> bool:
        """Check if PHI has been properly removed."""
        try:
            text_columns = df.select_dtypes(include=['object']).columns

            for col in text_columns:
                for pattern in self.phi_patterns.values():
                    if df[col].astype(str).str.contains(pattern, regex=True).any():
                        return False

            return True
        except Exception as e:
            self.logger.warning(f"PHI removal check failed: {str(e)}")
            return False

    def transform_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform raw data with feature engineering and cleaning."""
        self.logger.info("Starting data transformation")

        transformed_dfs = {}

        for table_name, df in dataframes.items():
            self.logger.info(f"Transforming {table_name}")

            # Make a copy for transformation
            df_transformed = df.copy()

            # Standardize datetime columns
            df_transformed = self._standardize_datetime_columns(df_transformed)

            # Remove PHI based on table type
            phi_columns = self._get_phi_columns_for_table(table_name)
            if phi_columns:
                df_transformed = self.remove_phi(df_transformed, phi_columns)

            # Apply table-specific transformations
            df_transformed = self._apply_table_specific_transforms(df_transformed, table_name)

            transformed_dfs[table_name] = df_transformed

        self.logger.info("Data transformation completed")
        return transformed_dfs

    def _standardize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize datetime columns."""
        date_columns = [col for col in df.columns if any(
            date_word in col.lower() 
            for date_word in ['date', 'start', 'stop', 'birth']
        )]

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def _get_phi_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns that contain PHI for a specific table."""
        phi_column_mapping = {
            'patients': ['SSN', 'DRIVERS', 'PASSPORT', 'ADDRESS'],
            'providers': ['ADDRESS'],
            'organizations': ['ADDRESS'],
            'encounters': [],  # Usually no direct PHI in encounters
            'observations': [],
            'conditions': [],
            'procedures': [],
            'medications': [],
            'allergies': [],
            'careplans': [],
            'immunizations': [],
            'devices': [],
            'supplies': [],
            'payers': [],
            'payer_transitions': []
        }

        return phi_column_mapping.get(table_name.lower(), [])

    def _apply_table_specific_transforms(self, df: pd.DataFrame, 
                                       table_name: str) -> pd.DataFrame:
        """Apply transformations specific to each table."""
        if table_name.lower() == 'encounters':
            # Add encounter duration
            if 'START' in df.columns and 'STOP' in df.columns:
                df['ENCOUNTER_DURATION_HOURS'] = (
                    pd.to_datetime(df['STOP']) - pd.to_datetime(df['START'])
                ).dt.total_seconds() / 3600

            # Add time-based features
            if 'START' in df.columns:
                start_dt = pd.to_datetime(df['START'])
                df['HOUR_OF_DAY'] = start_dt.dt.hour
                df['DAY_OF_WEEK'] = start_dt.dt.dayofweek
                df['MONTH'] = start_dt.dt.month
                df['YEAR'] = start_dt.dt.year

        elif table_name.lower() == 'patients':
            # Calculate age
            if 'BIRTHDATE' in df.columns:
                birth_dt = pd.to_datetime(df['BIRTHDATE'])
                df['AGE_YEARS'] = (datetime.now() - birth_dt).dt.days / 365.25

        return df

    def load_to_parquet(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """Load transformed data to Parquet format."""
        self.logger.info("Starting data loading to Parquet format")

        parquet_paths = {}
        parquet_dir = self.processed_path / "parquet"

        for table_name, df in dataframes.items():
            try:
                file_path = parquet_dir / f"{table_name}.parquet"

                # Convert to PyArrow table for better type handling
                table = pa.Table.from_pandas(df)

                # Write with compression
                pq.write_table(
                    table,
                    file_path,
                    compression='snappy',
                    write_statistics=True
                )

                parquet_paths[table_name] = file_path
                self.logger.info(f"Saved {table_name} to {file_path}")

            except Exception as e:
                self.logger.error(f"Error saving {table_name} to Parquet: {str(e)}")
                raise

        return parquet_paths

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ETL pipeline."""
        self.logger.info("Starting ETL pipeline execution")

        try:
            # Extract
            raw_dataframes = self.extract_synthea_data()

            # Transform
            transformed_dataframes = self.transform_data(raw_dataframes)

            # Validate quality
            quality_metrics = {}
            for table_name, df in transformed_dataframes.items():
                quality_metrics[table_name] = self.validate_data_quality(df, table_name)

            # Load
            parquet_paths = self.load_to_parquet(transformed_dataframes)

            pipeline_result = {
                "status": "success",
                "tables_processed": len(transformed_dataframes),
                "total_records": sum(len(df) for df in transformed_dataframes.values()),
                "parquet_paths": parquet_paths,
                "quality_metrics": quality_metrics,
                "processing_timestamp": datetime.now()
            }

            self.logger.info("ETL pipeline completed successfully")
            return pipeline_result

        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pipeline
    pipeline = ETLPipeline()
    result = pipeline.run_pipeline()
    print(f"Pipeline completed: {result['status']}")
