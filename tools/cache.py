import os

import redis

_client = None


def _get_client() -> redis.Redis | None:
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if url:
            _client = redis.from_url(url, decode_responses=True)
    return _client


def cache_get(key: str) -> str | None:
    r = _get_client()
    if r is None:
        return None
    try:
        return r.get(f"finance:{key}")
    except Exception:
        return None


def cache_set(key: str, value: str, ttl: int = 86400) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.set(f"finance:{key}", value, ex=ttl)
    except Exception:
        pass
