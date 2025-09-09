# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt
RUN pip install --no-cache-dir --timeout=600 -r requirements-dev.txt

# Copy only the source code to minimize rebuilds
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy application code
COPY data/ ./data/ 
COPY pyproject.toml .
COPY README.md .


# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
