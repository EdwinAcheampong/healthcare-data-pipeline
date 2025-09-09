"""
Orchestration script to run the entire healthcare data pipeline.

This script runs the following steps in order:
1. ETL pipeline
2. ML model execution
3. RL system execution
4. Tests

It also pushes metrics about its execution to the Prometheus Pushgateway.
"""

import subprocess
import logging
import sys
import time
import socket
import os
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- Prometheus Metrics Setup ---
PUSHGATEWAY_ADDRESS = os.getenv("PUSHGATEWAY_ADDRESS", "pushgateway:9091")
JOB_NAME = "healthcare_pipeline_batch"

# It's good practice to use a registry specific to this job
registry = CollectorRegistry()

g_last_success = Gauge(
    "job_last_success_timestamp_seconds",
    "Timestamp of the last successful completion of the job",
    registry=registry
)
g_last_failure = Gauge(
    "job_last_failure_timestamp_seconds",
    "Timestamp of the last failed completion of the job",
    registry=registry
)
g_last_duration = Gauge(
    "job_last_duration_seconds",
    "The duration of the last run of the job in seconds",
    registry=registry
)

# --- Logging Setup ---
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
    """
    Runs a command, streams its output in real-time, and waits for it to complete.
    
    Args:
        command: The command to run as a list of strings.
        
    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
    """
    logging.info(f"Running command: {' '.join(command)}")
    
    try:
        # Using subprocess.run with capture_output=False to stream output directly
        # to the parent's stdout/stderr, which are captured by our logger.
        # This is simpler and less prone to deadlocks than PIPE.
        result = subprocess.run(
            command,
            check=True,       # Raise CalledProcessError on non-zero exit codes
            text=True,        # Decode stdout/stderr as text
            encoding='utf-8',
            errors='replace'
        )
        logging.info(f"Command completed successfully with exit code {result.returncode}")

    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}. Please ensure it is in your PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        # No need to log stdout/stderr here as it's already streamed
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {e}")
        raise


def main():
    """Runs the entire pipeline and pushes metrics."""
    start_time = time.time()
    
    try:
        logging.info("Starting the healthcare data pipeline orchestration script.")

        # 1. Run ETL pipeline
        logging.info("----- Step 1: Running ETL pipeline -----")
        run_command([sys.executable, "src/data_pipeline/etl.py"])

        # 2. Run ML model execution
        logging.info("----- Step 2: Running ML model execution -----")
        run_command([sys.executable, "scripts/ml_model_execution.py"])

        # 3. Run RL system execution
        logging.info("----- Step 3: Running RL system execution -----")
        run_command([sys.executable, "scripts/rl_system_execution.py"])

        # 4. Run tests
        logging.info("----- Step 4: Running tests -----")
        run_command([sys.executable, "-m", "pytest", "tests/"])

        logging.info("All steps completed successfully!")
        g_last_success.set_to_current_time()

    except (subprocess.CalledProcessError, Exception) as e:
        logging.error(f"Pipeline execution failed: {e}")
        g_last_failure.set_to_current_time()
        sys.exit(1)
    
    finally:
        duration = time.time() - start_time
        g_last_duration.set(duration)
        
        logging.info(f"Pushing metrics to Pushgateway at {PUSHGATEWAY_ADDRESS}")
        try:
            push_to_gateway(
                PUSHGATEWAY_ADDRESS,
                job=JOB_NAME,
                registry=registry,
                grouping_key={'instance': socket.gethostname()}
            )
            logging.info("Successfully pushed metrics.")
        except Exception as e:
            logging.error(f"Could not push metrics to Pushgateway: {e}")

if __name__ == "__main__":
    main()