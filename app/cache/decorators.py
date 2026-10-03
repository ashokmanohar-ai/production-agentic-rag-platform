from app.agent.context import RuntimeContext
from app.cache.redis_cache import RedisCache
from app.llm.base import LLMProvider
from app.observability.langfuse import LangfuseObservability
from app.retrieval.base import Retriever


class CachedRetriever:
    def __init__(
        self,
        retriever: Retriever,
        cache: RedisCache,
        observability: LangfuseObservability,
    ) -> None:
        self.retriever = retriever
        self.cache = cache
        self.observability = observability

    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        payload: dict[str, object] = {
            "query": query,
            "top_k": context.top_k,
            "hybrid": context.use_hybrid,
            "categories": list(context.categories),
        }
        key = self.cache.key("retrieval", payload)
        cached = await self.cache.get_json(key)
        cached_items = cached.get("items") if cached else None
        if isinstance(cached_items, list):
            return [dict(item) for item in cached_items if isinstance(item, dict)]
        with self.observability.observation(
            "retrieve", "retriever", input_data=payload, metadata={"cache_hit": False}
        ) as span:
            items = await self.retriever.search(query, context)
            await self.cache.set_json(key, {"items": items})
            if span:
                span.update(output={"count": len(items)})
            return items


class CachedLLMProvider:
    def __init__(
        self,
        llm: LLMProvider,
        cache: RedisCache,
        observability: LangfuseObservability,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.observability = observability

    async def generate(self, prompt: str, model: str) -> str:
        key = self.cache.key("llm", {"prompt": prompt, "model": model})
        cached = await self.cache.get_json(key)
        cached_text = cached.get("text") if cached else None
        if isinstance(cached_text, str):
            return cached_text
        with self.observability.observation(
            "llm-generate", "generation", input_data={"prompt": prompt}, model=model,
            metadata={"cache_hit": False},
        ) as span:
            text = await self.llm.generate(prompt, model)
            await self.cache.set_json(key, {"text": text})
            if span:
                span.update(output=text)
            return text
