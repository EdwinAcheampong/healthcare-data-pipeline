"""
Configuration settings for the healthcare data pipeline project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    debug: bool = Field(default=True)
    testing: bool = Field(default=False)
    environment: str = Field(default="development")

    # Database configuration settings
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/msc_project")
    mongo_uri: str = Field(default="mongodb://localhost:27017/healthcare_data")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # API configuration settings
    api_host: str = Field(default="localhost")
    api_port: int = Field(default=8000)
    api_secret_key: str = Field(default="development-secret-key")
    jwt_secret_key: str = Field(default="jwt-development-secret")
    cors_origins: list = Field(default=["http://localhost:3000", "http://localhost:8080"])
    allowed_hosts: list = Field(default=["localhost", "127.0.0.1"])

    # Data processing configuration settings
    synthea_data_path: Path = Field(default=Path("/app/data/synthea/"))
    processed_data_path: Path = Field(default=Path("/app/data/processed/"))
    raw_data_path: Path = Field(default=Path("/app/data/raw/"))

    # Machine learning configuration settings
    model_registry_path: Path = Field(default=Path("./models/"))
    experiment_tracking_uri: str = Field(default="./mlruns/")
    model_serving_port: int = Field(default=8001)

    # FHIR server configuration settings
    fhir_server_url: str = Field(default="http://localhost:8082/fhir")
    hl7_validation_enabled: bool = Field(default=True)
    phi_anonymization: bool = Field(default=True)

    # Logging configuration settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file_path: Optional[Path] = Field(default=Path("./logs/app.log"))

    # Cloud Storage (Optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    azure_storage_connection_string: Optional[str] = None
    azure_container_name: Optional[str] = None

    # Monitoring & Logging
    sentry_dsn: Optional[str] = None
    prometheus_port: int = 9090

    # Airflow Configuration (if using)
    airflow_home: Optional[str] = None
    airflow_webserver_port: int = 8080
    airflow_scheduler_heartbeat_sec: int = 5

    # Jupyter Configuration
    jupyter_port: int = 8888
    jupyter_token: Optional[str] = None

    # Gemini API Key
    gemini_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()