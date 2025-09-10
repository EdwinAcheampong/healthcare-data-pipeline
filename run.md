# End-to-End Solution Guide

This document provides a complete, step-by-step guide to set up, run, and monitor the entire Healthcare Data Pipeline solution using the production Docker environment.

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
*(This is the correct URL for your project.)*

---

### Step 2: Configure Environment Variables

The application uses environment variables for configuration. A template is provided.

1.  **Copy the example file:**
    ```bash
    cp env.example .env
    ```
2.  **Review the `.env` file:** Open the newly created `.env` file in a text editor. The default values are suitable for the Docker environment, but you can customize them if needed (e.g., changing the Grafana password).

---

### Step 3: Place Raw Data

The ETL pipeline is configured to look for raw Synthea data.

1.  Navigate to the `data/raw/` directory.
2.  Place your raw Synthea CSV files inside this directory.

*(For detailed information on acquiring and setting up the Synthea dataset, please refer to the `data/DATA_SETUP.md` guide.)*

---

### Step 4: Build and Run the Entire Stack

This single command will build the Docker images (if they don't exist) and start all services (API, database, Prometheus, Grafana, etc.) in the background.

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```
- `--build`: Forces a rebuild of the images to ensure all recent changes are included.
- `-d`: Runs the containers in detached mode (in the background).

To check the status of your running containers, you can use `docker-compose -f docker-compose.prod.yml ps`.

---

### Step 5: Execute the Data & ML Pipelines

With the services running, you now need to execute the main processing script *inside* the `api` container. This script will run the ETL, train the ML models, execute the RL system, and run the project's tests.

```bash
docker-compose -f docker-compose.prod.yml exec api python scripts/run_all.py
```
You will see detailed logs in your terminal as the script progresses through each step. This process may take several minutes to complete.

---

### Step 6: Access the Grafana Dashboard

Once the pipeline execution is complete, the system is fully operational and monitoring data is being collected.

1.  **Open your web browser** and navigate to:
    [**http://localhost:3001**](http://localhost:3001)

2.  **Log in:**
    -   **Username:** `admin`
    -   **Password:** `admin` (or whatever you set in your `.env` file)

3.  **View the Dashboard:** Navigate to the "Dashboards" section. The pre-configured "Healthcare API Monitoring" dashboard will be available to view live metrics from the API and the results of the pipeline execution batch job.

---

### Step 7: Access the Streamlit Dashboard

1.  **Open your web browser** and navigate to:
    [**http://localhost:8501**](http://localhost:8501)

2.  **Run the dashboard:**
    ```bash
    streamlit run src/api/dashboard.py
    ```

---

### Stopping the Application

To stop all running services, run the following command:

```bash
docker-compose -f docker-compose.prod.yml down
```