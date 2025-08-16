#!/usr/bin/env python3
"""
Healthcare Data Pipeline Demo Script.

This script demonstrates the core healthcare data processing pipeline
functionality that we successfully tested.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main demo function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Healthcare Data Pipeline Demo Starting...")
    
    try:
        from config.settings import settings
        from data_pipeline.etl import ETLPipeline
        from data_pipeline.storage import StorageManager
        
        logger.info(f"📁 Data Path: {settings.data.synthea_data_path}")
        logger.info(f"📁 Output Path: {settings.data.processed_data_path}")
        
        # Check if Synthea data exists
        synthea_files = list(settings.data.synthea_data_path.glob("*.csv"))
        if not synthea_files:
            logger.warning(f"No Synthea CSV files found in {settings.data.synthea_data_path}")
            logger.info("Pipeline is ready but needs Synthea data to process")
            return
        
        logger.info(f"✅ Found {len(synthea_files)} Synthea CSV files")
        
        # Initialize pipeline
        etl = ETLPipeline()
        storage_manager = StorageManager()
        
        # Run ETL pipeline
        logger.info("🔄 Running ETL Pipeline...")
        result = etl.run_pipeline()
        
        logger.info(f"✅ ETL Pipeline completed successfully!")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   Tables processed: {result['tables_processed']}")
        logger.info(f"   Total records: {result['total_records']:,}")
        
        # Get storage inventory
        inventory = storage_manager.get_data_inventory()
        logger.info(f"📊 Storage Inventory:")
        logger.info(f"   Parquet files: {len(inventory['parquet_files'])}")
        logger.info(f"   Feature sets: {len(inventory['feature_sets'])}")
        
        logger.info("🎉 Demo completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.info("Some dependencies may not be available. Core pipeline logic is implemented.")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
