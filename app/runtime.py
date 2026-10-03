from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

import httpx
from opensearchpy import OpenSearch
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import Settings


@dataclass
class DependencyStatus:
    status: str
    detail: str | None = None


class RuntimeDiagnostics:
    def __init__(self, settings: Settings, sessions: sessionmaker) -> None:
        self.settings = settings
        self.sessions = sessions

    async def check(self) -> dict[str, object]:
        checks = await asyncio.gather(
            self._postgres(), self._redis(), self._opensearch(), self._ollama()
        )
        names = ("postgres", "redis", "opensearch", "ollama")
        dependencies = {name: asdict(value) for name, value in zip(names, checks, strict=True)}
        ready = all(item.status == "ok" for item in checks)
        return {"status": "ready" if ready else "not_ready", "dependencies": dependencies}

    async def _postgres(self) -> DependencyStatus:
        try:
            with self.sessions() as session:
                session.execute(text("SELECT 1"))
            return DependencyStatus("ok")
        except Exception as exc:
            return DependencyStatus("error", type(exc).__name__)

    async def _redis(self) -> DependencyStatus:
        client = Redis.from_url(self.settings.redis_url)
        try:
            await client.ping()
            return DependencyStatus("ok")
        except Exception as exc:
            return DependencyStatus("error", type(exc).__name__)
        finally:
            await client.aclose()

    async def _opensearch(self) -> DependencyStatus:
        client = OpenSearch(hosts=[self.settings.opensearch_url])
        try:
            healthy = await asyncio.to_thread(client.ping)
            return DependencyStatus("ok" if healthy else "error", None if healthy else "ping_failed")
        except Exception as exc:
            return DependencyStatus("error", type(exc).__name__)

    async def _ollama(self) -> DependencyStatus:
        try:
            async with httpx.AsyncClient(timeout=self.settings.runtime_health_timeout_seconds) as client:
                response = await client.get(f"{self.settings.ollama_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                models = {
                    item.get("name")
                    for item in response.json().get("models", [])
                    if isinstance(item, dict)
                }
            required = {self.settings.default_model, self.settings.embedding_model}
            missing = sorted(required - models)
            if missing:
                return DependencyStatus("error", "missing_models:" + ",".join(missing))
            return DependencyStatus("ok")
        except Exception as exc:
            return DependencyStatus("error", type(exc).__name__)
