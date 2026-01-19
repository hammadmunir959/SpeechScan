import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "SpeechScan API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(BASE_DIR.parent / "ProcessedData/models")))
    
    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Security
    ALLOWED_ORIGINS_STR: str = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:8000,http://127.0.0.1:8000")
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(",")]
    
    # Safety
    MEDICAL_DISCLAIMER: str = (
        "Medical Disclaimer: This tool is for research purposes only. "
        "It is not a diagnostic tool. Consult a healthcare professional."
    )
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS: bool = os.getenv("JSON_LOGS", "true").lower() == "true"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Initialize global settings
settings = Settings()
