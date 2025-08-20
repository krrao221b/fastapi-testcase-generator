from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    debug: bool = False
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # OpenAI Configuration (secrets come from environment)
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://models.github.ai/inference"
    openai_model: str = "openai/gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-large"

    # Gemini Configuration (optional)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-pro"
    # Embedding fallback policy (to avoid mixing vector spaces)
    # Keep False unless you separate collections per provider
    allow_gemini_embedding_fallback: bool = False

    # ChromaDB Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "test_cases"
    # Local embedding fallback
    use_local_embeddings: bool = False
    local_embedding_model: str = "all-MiniLM-L6-v2"
    # Similarity-based storage control
    # If a most-similar case has a similarity score >= this value, do not store the newly generated test case.
    # Lowered to 0.75 to allow fewer duplicates to be treated as duplicates (more permissive saving).
    # NOTE: similarity is cosine in [0,1]; user requested 7.5 — interpreting as 0.75.
    skip_store_if_similar_score: float = 0.75

    # JIRA Integration (configure via environment)
    jira_base_url: Optional[str] = None
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None

    # Zephyr Integration (configure via environment)
    zephyr_base_url: Optional[str] = "https://eu.api.zephyrscale.smartbear.com/v2"
    zephyr_api_token: Optional[str] = None

    # Database Configuration
    database_url: str = "sqlite:///./data/testcases.db"

    # Security
    # Security
    secret_key: Optional[str] = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
