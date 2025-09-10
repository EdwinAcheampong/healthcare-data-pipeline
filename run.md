# End-to-End Solution Guide

This document provides a complete, step-by-step guide to set up, run, and monitor the entire Healthcare Data Pipeline solution.

## Production Environment (Docker)

This is the recommended way to run the entire application stack.

### Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Git:** For cloning the repository.
- **Docker and Docker Compose:** To build and run the containerized application stack.

---

### Step 1: Clone the Repository

Open your terminal, navigate to the directory where you want to store the project, and clone the repository:

```bash
git clone https://github.com/EdwinAcheampong/healthcare-data-pipeline.git
cd healthcare-data-pipeline
```

---

### Step 2: Configure Environment Variables

The application uses environment variables for configuration. A template is provided.

1.  **Copy the example file:**
    ```bash
    cp env.example .env
    ```
2.  **Review the `.env` file:** Open the newly created `.env` file in a text editor. The default values are suitable for the Docker environment.

---

### Step 3: Place Raw Data

The ETL pipeline is configured to look for raw Synthea data.

1.  Navigate to the `data/raw/` directory.
2.  Place your raw Synthea CSV files inside this directory.

*(For detailed information on acquiring and setting up the Synthea dataset, please refer to the `data/DATA_SETUP.md` guide.)*

---

### Step 4: Build and Run the Entire Stack

This single command will build the Docker images and start all services in the background.

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

---

### Step 5: Execute the Data & ML Pipelines

Execute the main processing script *inside* the `api` container.

```bash
docker-compose -f docker-compose.prod.yml exec api python scripts/run_all.py
```

---

### Step 6: Access the Grafana Dashboard

1.  **Open your web browser** and navigate to:
    [**http://localhost:3001**](http://localhost:3001)

2.  **Log in:**
    -   **Username:** `admin`
    -   **Password:** `admin`

---

### Step 7: Stopping the Application

```bash
docker-compose -f docker-compose.prod.yml down
```

---

## Local Development (Hybrid Mode)

This mode is ideal for API and model development. It runs the monitoring services (Prometheus, Grafana) in Docker, while the FastAPI application runs locally on your machine.

### Step 1: Start Monitoring Services

Run the following command to start the Grafana and Prometheus containers:

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Step 2: Install Dependencies and Run API

In a separate terminal, run the following commands to install the Python dependencies and start the local API server:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.main:app --reload
```

### Step 3: Access Services

- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 4: Stopping the Hybrid Stack

1.  **Stop the API:** Press `Ctrl+C` in the terminal where the API is running.
2.  **Stop the monitoring services:**
    ```bash
    docker-compose -f docker-compose.monitoring.yml down
    ```
