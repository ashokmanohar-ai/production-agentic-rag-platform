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
    retrieval_cache_version: str = "v1"
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    ingestion_chunk_size: int = 1200
    ingestion_chunk_overlap: int = 200
    max_upload_bytes: int = 20_000_000
    worker_poll_seconds: float = 2.0
    worker_busy_poll_seconds: float = 0.2
    worker_job_lease_seconds: int = 300
    runtime_health_timeout_seconds: float = 5.0
    ollama_timeout_seconds: float = 60.0
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    auth_enabled: bool = False
    api_key_sha256: str | None = None
    oidc_enabled: bool = False
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
