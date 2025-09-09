"""
Storage Systems for Healthcare Data Pipeline.

This module provides storage interfaces for Parquet files,
time-series databases, and feature stores.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from abc import ABC, abstractmethod

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from ..config.settings import settings
except ImportError:
    from config.settings import settings


class StorageInterface(ABC):
    """Abstract base class for storage systems."""

    @abstractmethod
    def write(self, data: Any, identifier: str, **kwargs) -> bool:
        """Write data to storage."""
        pass

    @abstractmethod
    def read(self, identifier: str, **kwargs) -> Any:
        """Read data from storage."""
        pass

    @abstractmethod
    def delete(self, identifier: str) -> bool:
        """Delete data from storage."""
        pass

    @abstractmethod
    def list_items(self) -> List[str]:
        """List all items in storage."""
        pass


class ParquetStorage(StorageInterface):
    """Parquet file storage system for analytical workloads."""

    def __init__(self, base_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path or settings.processed_data_path / "parquet"
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_path = self.base_path / "_metadata"
        self.metadata_path.mkdir(exist_ok=True)

    def write(self, data: pd.DataFrame, identifier: str,
              compression: str = 'snappy', **kwargs) -> bool:
        """Write DataFrame to Parquet with optional partitioning."""
        try:
            file_path = self.base_path / f"{identifier}.parquet"

            # Convert to PyArrow table for better control
            table = pa.Table.from_pandas(data)

            # Write with compression and statistics
            pq.write_table(
                table,
                file_path,
                compression=compression,
                write_statistics=True,
                use_dictionary=True,
                **kwargs
            )

            # Store metadata
            metadata = {
                "identifier": identifier,
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "created_at": datetime.now().isoformat(),
                "file_size_bytes": file_path.stat().st_size,
                "compression": compression
            }

            metadata_file = self.metadata_path / f"{identifier}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved {identifier} to Parquet: {data.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write {identifier} to Parquet: {str(e)}")
            return False

    def read(self, identifier: str, columns: Optional[List[str]] = None,
             **kwargs) -> pd.DataFrame:
        """Read DataFrame from Parquet with optional column filtering."""
        try:
            file_path = self.base_path / f"{identifier}.parquet"

            if not file_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {identifier}")

            # Read with optional filtering
            df = pd.read_parquet(
                file_path,
                columns=columns,
                **kwargs
            )

            self.logger.info(f"Loaded {identifier} from Parquet: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to read {identifier} from Parquet: {str(e)}")
            raise

    def delete(self, identifier: str) -> bool:
        """Delete Parquet file and metadata."""
        try:
            file_path = self.base_path / f"{identifier}.parquet"
            metadata_file = self.metadata_path / f"{identifier}_metadata.json"

            if file_path.exists():
                file_path.unlink()

            if metadata_file.exists():
                metadata_file.unlink()

            self.logger.info(f"Deleted {identifier} from Parquet storage")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete {identifier}: {str(e)}")
            return False

    def list_items(self) -> List[str]:
        """List all available Parquet files."""
        parquet_files = list(self.base_path.glob("*.parquet"))
        return [f.stem for f in parquet_files]

    def get_metadata(self, identifier: str) -> Dict[str, Any]:
        """Get metadata for a Parquet file."""
        metadata_file = self.metadata_path / f"{identifier}_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {}


class FeatureStore(StorageInterface):
    """Feature store for ML features with versioning and metadata."""

    def __init__(self, base_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path or settings.data.processed_data_path / "features"
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Feature versioning
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(exist_ok=True)

        # Feature metadata
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)

    def write(self, data: pd.DataFrame, identifier: str,
              description: str = "",
              feature_type: str = "batch",
              version: Optional[int] = None,
              **kwargs) -> bool:
        """Write features to store with versioning."""
        try:
            # Determine version
            if version is None:
                version = self._get_next_version(identifier)

            # File paths
            feature_file = self.versions_path / f"{identifier}_v{version}.parquet"

            # Save feature data
            data.to_parquet(feature_file, compression='snappy')

            # Save metadata
            metadata = {
                "feature_name": identifier,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "feature_type": feature_type,
                "shape": data.shape,
                "columns": list(data.columns),
                "file_path": str(feature_file)
            }

            metadata_file = self.metadata_path / f"{identifier}_v{version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved features {identifier} v{version}: {data.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write features {identifier}: {str(e)}")
            return False

    def read(self, identifier: str, version: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Read features from store."""
        try:
            # Get latest version if not specified
            if version is None:
                version = self._get_latest_version(identifier)

            feature_file = self.versions_path / f"{identifier}_v{version}.parquet"

            if not feature_file.exists():
                raise FileNotFoundError(f"Features not found: {identifier} v{version}")

            df = pd.read_parquet(feature_file)
            self.logger.info(f"Loaded features {identifier} v{version}: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to read features {identifier}: {str(e)}")
            raise

    def delete(self, identifier: str, version: Optional[int] = None) -> bool:
        """Delete features from store."""
        try:
            if version is None:
                # Delete all versions
                versions = self._get_all_versions(identifier)
                for v in versions:
                    self.delete(identifier, v)
                return True

            # Delete specific version
            feature_file = self.versions_path / f"{identifier}_v{version}.parquet"
            metadata_file = self.metadata_path / f"{identifier}_v{version}_metadata.json"

            if feature_file.exists():
                feature_file.unlink()

            if metadata_file.exists():
                metadata_file.unlink()

            self.logger.info(f"Deleted features {identifier} v{version}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete features {identifier}: {str(e)}")
            return False

    def list_items(self) -> List[str]:
        """List all feature sets."""
        feature_files = list(self.versions_path.glob("*_v*.parquet"))
        feature_names = set()
        for f in feature_files:
            name = f.stem.rsplit('_v', 1)[0]
            feature_names.add(name)
        return list(feature_names)

    def _get_next_version(self, identifier: str) -> int:
        """Get next version number for a feature set."""
        latest = self._get_latest_version(identifier)
        return latest + 1 if latest else 1

    def _get_latest_version(self, identifier: str) -> Optional[int]:
        """Get latest version number for a feature set."""
        try:
            pattern = f"{identifier}_v*.parquet"
            files = list(self.versions_path.glob(pattern))
            if not files:
                return None

            versions = []
            for f in files:
                try:
                    version_part = f.stem.split('_v')[-1]
                    versions.append(int(version_part))
                except ValueError:
                    continue

            return max(versions) if versions else None
        except:
            return None

    def _get_all_versions(self, identifier: str) -> List[int]:
        """Get all version numbers for a feature set."""
        try:
            pattern = f"{identifier}_v*.parquet"
            files = list(self.versions_path.glob(pattern))
            versions = []
            for f in files:
                try:
                    version_part = f.stem.split('_v')[-1]
                    versions.append(int(version_part))
                except ValueError:
                    continue
            return sorted(versions)
        except:
            return []


class StorageManager:
    """Unified storage manager for all storage systems."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize storage systems
        self.parquet = ParquetStorage()
        self.features = FeatureStore()

    def save_processed_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Save processed data to appropriate storage systems."""
        self.logger.info("Saving processed data to storage systems")

        results = {
            "parquet": {},
            "cache": {}
        }

        for table_name, df in dataframes.items():
            try:
                # Save to Parquet for analytical queries
                parquet_success = self.parquet.write(df, f"processed_{table_name}")
                results["parquet"][table_name] = parquet_success

                # Cache summary statistics
                cache_success = self._cache_summary_stats(df, table_name)
                results["cache"][table_name] = cache_success

            except Exception as e:
                self.logger.error(f"Failed to save {table_name}: {str(e)}")
                results["parquet"][table_name] = False
                results["cache"][table_name] = False

        return results

    def save_features(self, features: Dict[str, pd.DataFrame],
                     descriptions: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
        """Save engineered features to feature store."""
        results = {}

        for feature_name, df in features.items():
            description = descriptions.get(feature_name, "") if descriptions else ""
            success = self.features.write(df, feature_name, description=description)
            results[feature_name] = success

        return results

    def get_data_inventory(self) -> Dict[str, Any]:
        """Get inventory of all stored data."""
        inventory = {
            "parquet_files": self.parquet.list_items(),
            "feature_sets": self.features.list_items(),
            "timestamp": datetime.now().isoformat()
        }

        # Add detailed metadata
        inventory["parquet_metadata"] = {}
        for item in inventory["parquet_files"]:
            inventory["parquet_metadata"][item] = self.parquet.get_metadata(item)

        return inventory

    def _cache_summary_stats(self, df: pd.DataFrame, table_name: str) -> bool:
        """Cache summary statistics."""
        try:
            stats = {
                "shape": df.shape,
                "columns": list(df.columns),
                "null_counts": df.isnull().sum().to_dict(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "cached_at": datetime.now().isoformat()
            }

            # Add numeric statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

            # Save to file for now (could be Redis in production)
            cache_dir = self.parquet.base_path.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"{table_name}_stats.json"

            with open(cache_file, 'w') as f:
                json.dump(stats, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Failed to cache stats for {table_name}: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    storage_manager = StorageManager()

    # Create sample data
    sample_data = {
        "test_table": pd.DataFrame({
            "id": range(100),
            "value": np.random.randn(100),
            "category": np.random.choice(['A', 'B', 'C'], 100),
            "timestamp": pd.date_range('2023-01-01', periods=100, freq='H')
        })
    }

    # Save to storage
    results = storage_manager.save_processed_data(sample_data)
    print(f"Storage results: {results}")

    # Get inventory
    inventory = storage_manager.get_data_inventory()
    print(f"Data inventory: {inventory}")
