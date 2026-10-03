import pytest

from app.agent.context import RuntimeContext
from app.cache.decorators import CachedLLMProvider, CachedRetriever
from app.cache.redis_cache import RedisCache
from app.observability.langfuse import LangfuseObservability


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int):
        self.values[key] = value

    async def scan_iter(self, match: str, count: int):
        prefix = match.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key

    async def delete(self, key: str):
        return 1 if self.values.pop(key, None) is not None else 0


class Retriever:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query, context):
        self.calls += 1
        return [{"text": "cached evidence", "document_id": "d", "chunk_id": "c", "title": "t"}]


class LLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, model: str) -> str:
        self.calls += 1
        return "cached answer"


@pytest.mark.asyncio
async def test_retrieval_and_llm_cache_hits() -> None:
    cache = RedisCache(FakeRedis(), 300)
    obs = LangfuseObservability(False)
    retriever = Retriever()
    cached_retriever = CachedRetriever(retriever, cache, obs)
    context = RuntimeContext(
        top_k=3, use_hybrid=False, model="m", categories=(),
        max_retrieval_attempts=3, guardrail_threshold=70,
    )
    assert await cached_retriever.search("query", context)
    assert await cached_retriever.search("query", context)
    assert retriever.calls == 1

    llm = LLM()
    cached_llm = CachedLLMProvider(llm, cache, obs)
    assert await cached_llm.generate("prompt", "m") == "cached answer"
    assert await cached_llm.generate("prompt", "m") == "cached answer"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_namespace_invalidation() -> None:
    redis = FakeRedis()
    cache = RedisCache(redis, 300)
    await cache.set_json(cache.key("retrieval", {"q": 1}), {"items": []})
    await cache.set_json(cache.key("llm", {"q": 1}), {"text": "x"})
    assert await cache.invalidate_rag() == 2
    assert not redis.values
