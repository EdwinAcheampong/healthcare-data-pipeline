"""
Configuration settings for the healthcare data pipeline project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="postgresql://postgres:postgres@localhost:5432/msc_project")
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/msc_project")
    mongo_uri: str = Field(default="mongodb://localhost:27017/healthcare_data")
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    class Config:
        env_prefix = "DATABASE_"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    secret_key: str = Field(default="development-secret-key")
    jwt_secret_key: str = Field(default="jwt-development-secret")
    cors_origins: list = Field(default=["http://localhost:3000", "http://localhost:8080"])
    allowed_hosts: list = Field(default=["localhost", "127.0.0.1"])
    log_level: str = Field(default="INFO")
    
    class Config:
        env_prefix = "API_"


class DataSettings(BaseSettings):
    """Data processing configuration settings."""
    
    synthea_data_path: Path = Field(default=Path("./data/synthea/"))
    processed_data_path: Path = Field(default=Path("./data/processed/"))
    raw_data_path: Path = Field(default=Path("./data/raw/"))
    
    class Config:
        env_prefix = "DATA_"


class MLSettings(BaseSettings):
    """Machine learning configuration settings."""
    
    model_registry_path: Path = Field(default=Path("./models/"))
    experiment_tracking_uri: str = Field(default="./mlruns/")
    model_serving_port: int = Field(default=8001)
    
    class Config:
        env_prefix = "ML_"


class FHIRSettings(BaseSettings):
    """FHIR server configuration settings."""
    
    server_url: str = Field(default="http://localhost:8082/fhir")
    validation_enabled: bool = Field(default=True)
    phi_anonymization: bool = Field(default=True)
    
    class Config:
        env_prefix = "FHIR_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[Path] = Field(default=Path("./logs/app.log"))
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    debug: bool = Field(default=True)
    testing: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    data: DataSettings = DataSettings()
    ml: MLSettings = MLSettings()
    fhir: FHIRSettings = FHIRSettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
