"""Cross-Encoder re-ranking service.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-rank retrieved chunks
by relevance to the query.
"""

import logging
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Singleton cross-encoder re-ranker."""

    _instance: Optional["RerankerService"] = None

    def __new__(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        logger.info("Loading reranker model: %s", settings.reranker_model)
        self.model = CrossEncoder(settings.reranker_model)
        self._initialized = True

    def rerank(self, query: str, chunks: List[Dict], top_k: int | None = None) -> List[Dict]:
        """Re-rank chunks by cross-encoder relevance score.

        Args:
            query: User query string.
            chunks: List of chunk dicts (must have 'text' key).
            top_k: Number of top chunks to return (default from config).

        Returns:
            Top-K chunks sorted by cross-encoder score descending.
        """
        if not chunks:
            return []

        top_k = top_k or settings.top_k_rerank
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
