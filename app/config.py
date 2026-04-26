"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql://smartdoc:smartdoc@localhost:5433/smartdoc"

    # --- Redis ---
    redis_url: str = "redis://localhost:6380/0"

    # --- LLM ---
    llm_provider: str = "google"
    google_api_key: str = ""
    llm_model: str = "gemini-3.1-flash-lite-preview"

    # --- Embedding & Reranking ---
    embedding_model: str = "models/gemini-embedding-2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- FAISS ---
    faiss_index_path: str = "data/faiss/index.faiss"
    faiss_metadata_path: str = "data/faiss/metadata.pkl"

    # --- Upload ---
    upload_dir: str = "data/uploads"

    # --- Chunking ---
    chunk_size: int = 2000
    chunk_overlap: int = 400

    # --- Retrieval ---
    top_k_retrieval: int = 12
    top_k_rerank: int = 4
    crag_top_k: int = 8
    min_relevant_chunks: int = 1
    enable_hyde: bool = True
    enable_crag: bool = True
    enable_reranker: bool = True
    retrieval_neighbor_window: int = 1

    # --- Caching ---
    embedding_cache_ttl: int = 3600
    query_cache_ttl: int = 300

    # --- Memory ---
    memory_summary_interval: int = 6

    # --- Security ---
    rate_limit: str = "30/minute"


settings = Settings()
