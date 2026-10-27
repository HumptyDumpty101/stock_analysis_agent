import json
import os

import redis.asyncio as aioredis

_TTL = 3600  # 1 hour


def _client() -> aioredis.Redis:
    return aioredis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)


async def load_history(session_id: str) -> list[dict]:
    if not os.getenv("REDIS_URL"):
        return []
    r = _client()
    try:
        data = await r.get(f"session:{session_id}")
        return json.loads(data) if data else []
    finally:
        await r.aclose()


async def save_history(session_id: str, messages: list[dict]) -> None:
    if not os.getenv("REDIS_URL"):
        return
    r = _client()
    try:
        await r.set(f"session:{session_id}", json.dumps(messages), ex=_TTL)
    finally:
        await r.aclose()
