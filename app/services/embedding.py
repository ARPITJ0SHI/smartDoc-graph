"""Embedding service with Redis caching layer.

Singleton GoogleGenerativeAIEmbeddings wrapper. Embedding vectors are cached in Redis
using the SHA-256 hash of the input text as the key.
"""

import hashlib
import json
import logging
from typing import List, Optional

import numpy as np
import redis
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages text embeddings with optional Redis cache."""

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key
        )
        try:
            self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._redis.ping()
            self._cache_available = True
            logger.info("Embedding cache connected to Redis.")
        except Exception:
            self._redis = None
            self._cache_available = False
            logger.warning("Redis unavailable — embedding caching disabled.")
        self._initialized = True

    # ---- Cache helpers ----

    def _cache_key(self, text: str) -> str:
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        if not self._cache_available:
            return None
        try:
            data = self._redis.get(self._cache_key(text))
            if data is not None:
                return np.array(json.loads(data), dtype=np.float32)
        except Exception:
            pass
        return None

    def _set_cached(self, text: str, vector: np.ndarray) -> None:
        if not self._cache_available:
            return
        try:
            self._redis.setex(
                self._cache_key(text),
                settings.embedding_cache_ttl,
                json.dumps(vector.tolist()),
            )
        except Exception:
            pass

    # ---- Public API ----

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts, using cache where possible."""
        results = [None] * len(texts)
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        # Compute uncached embeddings safely
        if uncached_indices:
            import concurrent.futures

            def _embed_single(text: str) -> np.ndarray:
                """Embed a single string to avoid the gemini-embedding-2 multimodal list bug."""
                # Empty strings cause INVALID_ARGUMENT, so fallback to a space
                safe_text = text if text.strip() else " "
                emb = self.model.embed_query(safe_text)
                return np.array(emb, dtype=np.float32)

            # Use ThreadPoolExecutor to speed up sequential API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_idx = {
                    executor.submit(_embed_single, texts[idx]): idx 
                    for idx in uncached_indices
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    vec = future.result()
                    results[idx] = vec
                    self._set_cached(texts[idx], vec)

        return np.vstack(results)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        cached = self._get_cached(query)
        if cached is not None:
            return cached
        safe_query = query if query.strip() else " "
        embedding = self.model.embed_query(safe_query)
        vec = np.array(embedding, dtype=np.float32)
        self._set_cached(query, vec)
        return vec

