"""
Orchestration script to run the entire healthcare data pipeline.

This script runs the following steps in order:
1. ETL pipeline
2. ML model execution
3. RL system execution
4. Tests
"""

import subprocess
import logging
import sys
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "run_all.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_command(command: list[str]):
    """Runs a command and logs the output."""
    logging.info(f"Running command: {' '.join(command)}")
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if process.stdout:
        logging.info(process.stdout)
    if process.stderr:
        logging.error(process.stderr)
    
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, command)
    
    logging.info("Command completed successfully.")

def main():
    """Runs the entire pipeline."""
    try:
        logging.info("Starting the healthcare data pipeline orchestration script.")

        # 1. Run ETL pipeline
        logging.info("----- Step 1: Running ETL pipeline -----")
        run_command(["python", "src/data_pipeline/etl.py"])

        # 2. Run ML model execution
        logging.info("----- Step 2: Running ML model execution -----")
        run_command(["python", "scripts/ml_model_execution.py"])

        # 3. Run RL system execution
        logging.info("----- Step 3: Running RL system execution -----")
        run_command(["python", "scripts/rl_system_execution.py"])

        # 4. Run tests
        logging.info("----- Step 4: Running tests -----")
        run_command(["pytest", "tests/"])

        logging.info("All steps completed successfully!")

    except subprocess.CalledProcessError:
        logging.error("Pipeline execution failed.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
