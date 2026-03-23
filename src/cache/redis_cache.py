"""
Redis Caching Layer for ContractIQ
=====================================
Caches LLM responses to:
1. Reduce latency — same question on same doc returns instantly
2. Reduce API costs — no duplicate LLM calls
3. Persist RAG sessions across app restarts

Cache strategy:
- Key: SHA256 hash of (session_id + question + retriever_type)
- Value: JSON serialized response
- TTL: 24 hours by default (configurable)

Why this matters in production:
- LLM calls average 2-4 seconds
- Cached responses return in <50ms
- Same NDA uploaded by 100 users = 1 LLM call, not 100
"""

import os
import json
import hashlib
from typing import Any, Optional, Dict
from datetime import timedelta

import redis
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class RedisCache:
    """
    Redis cache client for ContractIQ.
    Handles connection, serialization, and cache operations.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 86400,  # 24 hours in seconds
        prefix: str = "contractiq",
    ):
        """
        Args:
            redis_url: Redis connection URL. Defaults to REDIS_URL env var.
            default_ttl: Cache expiry in seconds. Default 24 hours.
            prefix: Key prefix to namespace all ContractIQ keys.
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        """Establish Redis connection with error handling."""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self.client.ping()
            log.info("Redis connection established", url=self.redis_url)
        except redis.ConnectionError as e:
            log.warning(
                "Redis unavailable — caching disabled. App will work without cache.",
                error=str(e),
            )
            self.client = None
        except Exception as e:
            log.warning("Redis initialization failed — caching disabled", error=str(e))
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if Redis is connected and available."""
        if self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def _make_key(self, *parts: str) -> str:
        """
        Build a namespaced cache key.
        Uses SHA256 hash to keep keys short and safe.

        Args:
            *parts: String parts to combine into a key

        Returns:
            Formatted cache key: "contractiq:abc123def456..."
        """
        combined = ":".join(str(p) for p in parts)
        hashed = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{hashed}"

    def get(self, *key_parts: str) -> Optional[Dict]:
        """
        Retrieve a cached value.

        Args:
            *key_parts: Parts to build the cache key

        Returns:
            Deserialized dict if cache hit, None if miss or unavailable
        """
        if not self.is_available:
            return None

        try:
            key = self._make_key(*key_parts)
            value = self.client.get(key)

            if value is None:
                log.info("Cache miss", key=key[:20])
                return None

            log.info("Cache hit", key=key[:20])
            return json.loads(value)

        except Exception as e:
            log.warning("Cache get failed — proceeding without cache", error=str(e))
            return None

    def set(self, value: Any, *key_parts: str, ttl: Optional[int] = None) -> bool:
        """
        Store a value in cache.

        Args:
            value: Dict/list to cache (must be JSON serializable)
            *key_parts: Parts to build the cache key
            ttl: Override default TTL in seconds

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available:
            return False

        try:
            key = self._make_key(*key_parts)
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            expiry = ttl or self.default_ttl

            self.client.setex(key, timedelta(seconds=expiry), serialized)
            log.info("Cache set", key=key[:20], ttl=expiry)
            return True

        except Exception as e:
            log.warning("Cache set failed — proceeding without cache", error=str(e))
            return False

    def delete(self, *key_parts: str) -> bool:
        """
        Delete a cached value.

        Args:
            *key_parts: Parts to build the cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_available:
            return False

        try:
            key = self._make_key(*key_parts)
            self.client.delete(key)
            log.info("Cache key deleted", key=key[:20])
            return True
        except Exception as e:
            log.warning("Cache delete failed", error=str(e))
            return False

    def invalidate_session(self, session_id: str) -> int:
        """
        Delete all cached responses for a session.
        Call this when a user re-uploads documents.

        Args:
            session_id: Session to invalidate

        Returns:
            Number of keys deleted
        """
        if not self.is_available:
            return 0

        try:
            pattern = f"{self.prefix}:*"
            keys = self.client.keys(pattern)
            deleted = 0

            for key in keys:
                if session_id in key:
                    self.client.delete(key)
                    deleted += 1

            log.info("Session cache invalidated", session_id=session_id, keys_deleted=deleted)
            return deleted
        except Exception as e:
            log.warning("Session invalidation failed", error=str(e))
            return 0

    def store_rag_session(self, session_id: str, metadata: Dict, ttl: int = 86400) -> bool:
        """
        Store RAG session metadata in Redis so it survives app restarts.

        Args:
            session_id: Session identifier
            metadata: Dict with retriever_type, faiss_dir, k, etc.
            ttl: How long to keep session alive (default 24h)

        Returns:
            True if stored successfully
        """
        return self.set(metadata, "session", session_id, ttl=ttl)

    def get_rag_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve RAG session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata dict or None if expired/not found
        """
        return self.get("session", session_id)

    def get_stats(self) -> Dict:
        """
        Get cache statistics for monitoring.

        Returns:
            Dict with memory usage, hit rate, connected clients etc.
        """
        if not self.is_available:
            return {"status": "unavailable"}

        try:
            info = self.client.info()
            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global cache instance — import this everywhere
cache = RedisCache()
