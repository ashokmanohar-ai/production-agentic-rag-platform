from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Production Agentic RAG Platform"
    default_model: str = "llama3.2:3b"
    default_top_k: int = 3
    max_top_k: int = 10
    max_retrieval_attempts: int = 3
    guardrail_threshold: int = 70
    guardrail_fail_closed: bool = True
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag-chunks"
    opensearch_neural_model_id: str | None = None
    opensearch_vector_field: str = "embedding"
    opensearch_search_pipeline: str | None = None
    redis_url: str = "redis://localhost:6379/0"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    ingestion_chunk_size: int = 1200
    ingestion_chunk_overlap: int = 200
    max_upload_bytes: int = 20_000_000
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
