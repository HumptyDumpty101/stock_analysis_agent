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
    """Return a cached(key, fetch_fn) closure backed by a TTLCache.

    Each call creates an independent cache — callers get their own TTL.
    Financial data: ttl=86_400 (24 h). News: ttl=3_600 (1 h).
    """
    _cache = TTLCache(maxsize=100, ttl=ttl)
    _lock  = Lock()

    def _cached(key: str, fetch_fn):
        with _lock:
            if key in _cache:
                return _cache[key]
        result = fetch_fn()
        with _lock:
            _cache[key] = result
        return result

    return _cached
