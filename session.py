import json
import os

import redis.asyncio as aioredis

_TTL = 3600  # 1 hour


def _client() -> aioredis.Redis:
    return aioredis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)


async def load_session(session_id: str) -> dict:
    """Returns {"messages": [...], "pending_ticker": str|None, "pending_name": str|None}"""
    if not os.getenv("REDIS_URL"):
        return {"messages": [], "pending_ticker": None, "pending_name": None}
    r = _client()
    try:
        data = await r.get(f"session:{session_id}")
        if not data:
            return {"messages": [], "pending_ticker": None, "pending_name": None}
        return json.loads(data)
    finally:
        await r.aclose()


async def save_session(session_id: str, data: dict) -> None:
    """data should have keys: messages, pending_ticker (optional), pending_name (optional)"""
    if not os.getenv("REDIS_URL"):
        return
    r = _client()
    try:
        await r.set(f"session:{session_id}", json.dumps(data), ex=_TTL)
    finally:
        await r.aclose()
