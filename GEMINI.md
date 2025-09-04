# GEMINI.md

## Project Overview

This project is a comprehensive healthcare data pipeline that uses machine learning and reinforcement learning to optimize healthcare workloads. It processes synthetic healthcare data (Synthea) to predict patient volumes and optimize resource allocation while adhering to healthcare compliance standards.

The main technologies used are:

*   **Python:** The core language for the entire pipeline.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn, XGBoost, Statsmodels, Prophet:** For machine learning and forecasting.
*   **PyTorch:** For the reinforcement learning (PPO) agent.
*   **FastAPI:** For serving the models via a REST API.
*   **Docker:** For containerizing the application and its services.
*   **PostgreSQL, MongoDB, Redis:** As data and caching layers.
*   **Jupyter:** For data exploration and notebooks.

## Building and Running

### Using Docker (Recommended)

The easiest way to get the entire system running is with Docker Compose.

```bash
# Start all services in the background
docker-compose up -d

# Stop all services
docker-compose down
```

### Local Development

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the ETL pipeline:**

    ```bash
    python src/data_pipeline/etl.py
    ```

3.  **Run the ML pipeline:**

    ```bash
    python scripts/ml_model_execution.py
    ```

4.  **Run the RL pipeline:**

    ```bash
    python scripts/rl_system_execution.py
    ```

5.  **Start the API server:**

    ```bash
    uvicorn src.api.main:app --reload
    ```

### Testing

Run the test suite using pytest:

```bash
pytest
```

## Development Conventions

*   **Code Style:** The project uses `black` for code formatting and `isort` for import sorting. The configuration is in `pyproject.toml`.
*   **Linting:** `flake8` and `mypy` are used for linting and type checking. The configuration is in `pyproject.toml`.
*   **Testing:** Tests are written using `pytest`. Test files are located in the `tests` directory and are separated into `unit` and `integration` tests. Fixtures are defined in `tests/conftest.py`.

## Key Files

*   `README.md`: The main entry point for understanding the project.
*   `pyproject.toml`: Defines project metadata, dependencies, and tool configurations.
*   `docker-compose.yml`: Defines the services for the Dockerized environment.
*   `Makefile`: Contains convenient shortcuts for common commands.
*   `src/main.py`: A CLI entry point for the application.
*   `src/data_pipeline/etl.py`: The main script for the ETL pipeline.
*   `src/models/baseline_models.py`: Implements baseline machine learning models.
*   `src/models/advanced_models.py`: Implements advanced machine learning models.
*   `src/models/ppo_agent.py`: Implements the PPO reinforcement learning agent.
*   `scripts/ml_model_execution.py`: Orchestrates the execution of the machine learning pipeline.
*   `scripts/rl_system_execution.py`: Orchestrates the execution of the reinforcement learning pipeline.
*   `tests/conftest.py`: Contains pytest fixtures and configuration for tests.

## Directory Structure

```
healthcare-data-pipeline/
├── data/                    # Healthcare data files
├── docs/                    # Complete documentation
├── logs/                    # Application logs
├── metrics/                 # Performance metrics
├── models/                  # Trained ML models
├── notebooks/               # Jupyter notebooks
├── reports/                 # Generated reports
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   ├── api/                 # API endpoints
│   ├── config/              # Configuration management
│   ├── data_pipeline/       # ETL and data processing
│   ├── models/              # ML and RL models
│   └── utils/               # Utility functions
├── tests/                   # Test files
├── docker-compose.yml       # Development setup
├── Dockerfile               # Application container
├── README.md                # Project README
└── requirements.txt         # Dependencies
```
