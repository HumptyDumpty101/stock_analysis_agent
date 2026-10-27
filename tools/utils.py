import json
import math
from threading import Lock

from cachetools import TTLCache


def _safe_float(v):
    """Anything → float, or None if it isn't a usable number."""
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _millions(v):
    """Raw dollars → millions: 383285000000 → 383285.0"""
    f = _safe_float(v)
    return round(f / 1_000_000, 1) if f is not None else None


def _extract_row(df, *names, col):
    """Value of the first row name that exists in the DataFrame (in millions)."""
    for name in names:
        if name in df.index:
            return _millions(df.loc[name, col])
    return None


def make_cached(ttl: int = 86_400):
    """Return a cached(key, fetch_fn, cache_only) closure.

    Checks Redis first, falls back to in-memory TTLCache, then fetches from API.
    If cache_only=True, returns an error dict instead of hitting the API.
    Financial data: ttl=86_400 (24 h). News: ttl=3_600 (1 h).
    """
    _mem = TTLCache(maxsize=100, ttl=ttl)
    _lock = Lock()

    def _cached(key: str, fetch_fn, cache_only: bool = False) -> str:
        from .cache import cache_get, cache_set

        # 1. Redis
        val = cache_get(key)
        if val:
            return val

        # 2. In-memory
        with _lock:
            if key in _mem:
                return _mem[key]

        # 3. cache_only — don't hit the API
        if cache_only:
            return json.dumps({"error": f"No cached data for {key}. Run a full analysis first."})

        # 4. Fetch from API then populate both caches
        result = fetch_fn()
        cache_set(key, result, ttl=ttl)
        with _lock:
            _mem[key] = result
        return result

    return _cached
