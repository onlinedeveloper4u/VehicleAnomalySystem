"""
Centralized configuration management using Pydantic Settings.

Environment variables take precedence over defaults.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_key: str = ""
    api_title: str = "Vehicle Anomaly Detection System"
    api_version: str = "1.0"
    
    # Model Configuration
    model_version: str = "v1"
    model_dir: str = "models"
    
    # Rate Limiting
    rate_limit: str = "100/minute"
    
    # Payload Limits
    max_records_per_request: int = 1000
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "api_inference.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
