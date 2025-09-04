# How to Run This Solution

This document provides a step-by-step guide to get the Healthcare Data Analytics & ML Pipeline up and running.

## 1. Prerequisites

Make sure you have the following software installed on your system:

- **Git:** For cloning the repository.
- **Python 3.9+:** For running the application and scripts.
- **Docker and Docker Compose:** For running the application in a containerized environment (recommended).

## 2. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/healthcare-data-pipeline.git
cd healthcare-data-pipeline
```

## 3. Environment Setup

It is recommended to use a Python virtual environment to manage dependencies.

### Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Configure Environment Variables

Copy the example environment file and modify it if necessary.

```bash
cp env.example .env
```

## 4. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, you can use the `Makefile` command:

```bash
make install
```

## 5. Running with Docker (Recommended)

Using Docker Compose is the recommended way to run the entire application stack, including the database, cache, and other services.

```bash
docker-compose up
```

This command will build the Docker images and start all the services defined in `docker-compose.yml`.

To run in detached mode (in the background), use:

```bash
docker-compose up -d
```

You can also use the `Makefile` for convenience:

```bash
make docker-up
```

## 6. Running the ETL Pipeline

The ETL (Extract, Transform, Load) pipeline processes the raw Synthea data, cleans it, removes PHI, and saves it in Parquet format.

To run the ETL pipeline, execute the following script:

```bash
python src/data_pipeline/etl.py
```

Alternatively, you can use the `Makefile` command which points to a similar script:

```bash
make process-data
```

## 7. Running the Machine Learning Pipeline

This will train the baseline and advanced machine learning models for patient volume prediction.

```bash
python scripts/ml_model_execution.py
```

Reports and model artifacts will be saved to the `reports` and `models` directories, respectively.

## 8. Running the Reinforcement Learning Pipeline

This will execute the reinforcement learning system for healthcare workload optimization.

```bash
python scripts/rl_system_execution.py
```

Execution reports and summaries will be saved in the `reports` directory.

## 9. Running Tests

To ensure everything is working correctly, run the test suite:

```bash
pytest
```

Or with the `Makefile`:

```bash
make test
```

## 10. Starting the API Server

To start the FastAPI server for predictions and optimizations:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with the `Makefile`:

```bash
make serve
```

The API documentation will be available at `http://localhost:8000/docs`.

## 11. Using Jupyter Notebooks

For data exploration and analysis, you can use Jupyter Lab.

```bash
jupyter lab
```

Or with the `Makefile`:

```bash
make jupyter
```

This will start the Jupyter Lab server, and you can access it in your browser at the provided URL (usually `http://localhost:8888`).
