import hashlib
import json
from redis.asyncio import Redis


class RedisCache:
    def __init__(self, redis: Redis, ttl_seconds: int = 300) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(namespace: str, payload: dict[str, object]) -> str:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return f"agentic-rag:{namespace}:{digest}"

    async def get_json(self, key: str) -> dict[str, object] | None:
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value: dict[str, object]) -> None:
        await self.redis.set(key, json.dumps(value), ex=self.ttl_seconds)
