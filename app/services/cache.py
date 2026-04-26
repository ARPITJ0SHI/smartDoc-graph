"""Redis-backed query-result cache.

Sits at the start of the chat endpoint (before LangGraph) and at the end
(after generate_answer). Skipped if session has new messages since cache write.
"""

import hashlib
import json
import logging
from typing import Optional
from uuid import UUID

import redis

from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Thin Redis wrapper for query → response caching."""

    _instance: Optional["CacheService"] = None

    def __new__(cls) -> "CacheService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        try:
            self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._redis.ping()
            self._available = True
            logger.info("Query cache connected to Redis.")
        except Exception:
            self._redis = None
            self._available = False
            logger.warning("Redis unavailable — query caching disabled.")
        self._initialized = True

    @staticmethod
    def _make_key(query: str, session_id: UUID) -> str:
        raw = f"{query}:{session_id}"
        return f"qcache:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get_cached_response(self, query: str, session_id: UUID) -> Optional[dict]:
        """Return cached response dict or None."""
        if not self._available:
            return None
        try:
            data = self._redis.get(self._make_key(query, session_id))
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def set_cached_response(self, query: str, session_id: UUID, response: dict) -> None:
        """Cache a response dict with TTL."""
        if not self._available:
            return
        try:
            self._redis.setex(
                self._make_key(query, session_id),
                settings.query_cache_ttl,
                json.dumps(response),
            )
        except Exception:
            pass
