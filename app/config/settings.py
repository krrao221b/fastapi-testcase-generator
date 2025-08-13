from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    debug: bool = False
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # OpenAI Configuration
    openai_api_key: str = "Add_Your_API_Key_Guys"
    openai_base_url: str = "https://models.github.ai/inference"
    openai_model: str = "openai/gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-large"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "test_cases"
    
    # JIRA Integration
    jira_base_url: Optional[str] = "https://testcaseready.atlassian.net"
    jira_username: Optional[str] = "achinthyabangera1@gmail.com"
    jira_api_token: Optional[str] = "ATATT3xFfGF05XQMnQE9XnwN9Bdvv4XW6r-cvevXE3aZwZ7trpybe86RIvRgfTw5c5P-FQzxzwGBRTnkQPfff6tl0QVGd6KWQy9JpfLtUmgXG0rltq_icjfSM0fbEbypS6eSisJXMx9Z2A0y0I7kUpdtRqO03YKrWQV-KsXGZsR8ask-hNIQwCQ=5B99F483"
    
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
