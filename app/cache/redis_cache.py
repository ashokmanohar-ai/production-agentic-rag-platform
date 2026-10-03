import hashlib
import json

from redis.asyncio import Redis


class RedisCache:
    def __init__(self, redis: Redis, ttl_seconds: int = 300) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(namespace: str, payload: dict[str, object]) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"agentic-rag:{namespace}:{digest}"

    async def get_json(self, key: str) -> dict[str, object] | None:
        try:
            raw = await self.redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_json(self, key: str, value: dict[str, object]) -> None:
        try:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_seconds)
        except Exception:
            return

    async def invalidate_namespace(self, namespace: str) -> int:
        pattern = f"agentic-rag:{namespace}:*"
        deleted = 0
        try:
            async for key in self.redis.scan_iter(match=pattern, count=100):
                deleted += int(await self.redis.delete(key))
        except Exception:
            return deleted
        return deleted

    async def invalidate_rag(self) -> int:
        return (
            await self.invalidate_namespace("retrieval")
            + await self.invalidate_namespace("llm")
        )
