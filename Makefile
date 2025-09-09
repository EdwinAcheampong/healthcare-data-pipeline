# MSc Healthcare Project Makefile

.PHONY: help setup install test clean lint format docker-up docker-down jupyter

# Default target
help:
	@echo "Available commands:"
	@echo "  setup      - Set up the development environment"
	@echo "  install    - Install Python dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  docker-up  - Start Docker services"
	@echo "  docker-down- Stop Docker services"
	@echo "  jupyter    - Start Jupyter Lab"
	@echo "  clean      - Clean temporary files"

# Environment setup
setup: install docker-up
	@echo "Setting up project directories..."
	@mkdir -p data/{raw,processed,synthea} logs models mlruns
	@touch data/raw/.gitkeep data/processed/.gitkeep
	@if [ ! -f .env ]; then cp env.example .env; fi
	@echo "Setup complete!"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	@pytest tests/ -v --cov=src

# Run linting
lint:
	@echo "Running linting..."
	@flake8 src/ tests/
	@mypy src/

# Format code
format:
	@echo "Formatting code..."
	@black src/ tests/
	@isort src/ tests/

# Start Docker services
docker-up:
	@echo "Starting Docker services..."
	@docker-compose up -d

# Start only essential Docker services for faster development
docker-up-fast:
	@echo "Starting essential Docker services only..."
	@docker-compose -f docker-compose.dev.yml up -d

# Start all services with profiles
docker-up-full:
	@echo "Starting all Docker services..."
	@docker-compose --profile full up -d

# Stop Docker services
docker-down:
	@echo "Stopping Docker services..."
	@docker-compose down

# Stop fast development services
docker-down-fast:
	@echo "Stopping fast Docker services..."
	@docker-compose -f docker-compose.dev.yml down

# Stop all services
docker-down-all:
	@echo "Stopping all Docker services..."
	@docker-compose down
	@docker-compose -f docker-compose.dev.yml down

# Start Jupyter Lab
jupyter:
	@echo "Starting Jupyter Lab..."
	@jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .coverage htmlcov/ .pytest_cache/

# Development shortcuts
dev: setup
	@echo "Development environment ready!"

# Production build
build:
	@echo "Building production image..."
	@docker build -t msc-healthcare-project .

# Database migration (placeholder)
migrate:
	@echo "Running database migrations..."
	@python scripts/migrate_db.py

# Data processing
process-data:
	@echo "Processing Synthea data..."
	@python src/data_pipeline/process_synthea.py

# Model training
train:
	@echo "Training models..."
	@python src/models/train.py

# API server
serve:
	@echo "Starting API server..."
	@uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

