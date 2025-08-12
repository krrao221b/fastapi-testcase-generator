from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    debug: bool = False
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 4200
    api_prefix: str = "/api/v1"
    
    # Gemini Configuration
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "test_cases"
    
    # JIRA Integration
    jira_base_url: Optional[str] = None
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    
    # Zephyr Integration
    zephyr_base_url: Optional[str] = None
    zephyr_access_key: Optional[str] = None
    zephyr_secret_key: Optional[str] = None
    
    # Database Configuration
    database_url: str = "sqlite:///./data/testcases.db"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
