import asyncio
from typing import Any, Protocol, cast

from app.agent.context import RuntimeContext
from app.embeddings.base import EmbeddingProvider


class SearchClient(Protocol):
    def search(
        self, *, index: str, body: dict[str, object], params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        ...


class OpenSearchRetriever:
    """Tenant-scoped BM25 retrieval with application-side BM25 + KNN RRF fusion."""

    def __init__(
        self,
        client: SearchClient,
        embeddings: EmbeddingProvider | None = None,
        index: str = "rag-chunks",
        vector_field: str = "embedding",
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.index = index
        self.vector_field = vector_field
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def _filters(self, context: RuntimeContext) -> list[dict[str, object]]:
        filters: list[dict[str, object]] = [
            {"term": {"tenant_id": context.tenant_id}},
            {"term": {"project_id": context.project_id}},
        ]
        if context.categories:
            filters.append({"terms": {"category": list(context.categories)}})
        return filters

    async def _run(self, body: dict[str, object]) -> list[dict[str, object]]:
        response = await asyncio.to_thread(
            self.client.search, index=self.index, body=body, params=None
        )
        return list(response.get("hits", {}).get("hits", []))

    @staticmethod
    def _source(hit: dict[str, object], mode: str, score: float | None = None) -> dict[str, object]:
        source = dict(cast(dict[str, object], hit.get("_source", {})))
        raw_score = hit.get("_score")
        source["score"] = float(cast(float | int, raw_score or 0.0)) if score is None else score
        source["search_mode"] = mode
        return source

    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        filters = self._filters(context)
        lexical_body: dict[str, object] = {
            "size": context.top_k if not context.use_hybrid else context.top_k * self.candidate_multiplier,
            "query": {"bool": {"must": [{"match": {"text": query}}], "filter": filters}},
        }
        lexical_hits = await self._run(lexical_body)
        if not context.use_hybrid:
            return [self._source(hit, "bm25") for hit in lexical_hits[: context.top_k]]

        if self.embeddings is None:
            raise RuntimeError("Hybrid search requires the configured query embedding provider")
        vectors = await self.embeddings.embed([query])
        if len(vectors) != 1:
            raise RuntimeError("Embedding provider did not return one query vector")
        vector = vectors[0]
        if len(vector) != self.embeddings.dimensions:
            raise RuntimeError(
                f"Query embedding dimension {len(vector)} does not match configured "
                f"dimension {self.embeddings.dimensions}"
            )

        candidates = context.top_k * self.candidate_multiplier
        vector_body: dict[str, object] = {
            "size": candidates,
            "query": {
                "knn": {
                    self.vector_field: {
                        "vector": vector,
                        "k": candidates,
                        "filter": {"bool": {"filter": filters}},
                    }
                }
            },
        }
        vector_hits = await self._run(vector_body)

        ranked: dict[str, tuple[dict[str, object], float]] = {}
        for hits in (lexical_hits, vector_hits):
            for rank, hit in enumerate(hits, start=1):
                source = dict(cast(dict[str, object], hit.get("_source", {})))
                identity = str(source.get("chunk_id") or hit.get("_id") or "")
                if not identity:
                    continue
                previous = ranked.get(identity)
                score = (previous[1] if previous else 0.0) + 1.0 / (self.rrf_k + rank)
                ranked[identity] = (hit if previous is None else previous[0], score)

        ordered = sorted(ranked.values(), key=lambda item: (-item[1], str(item[0].get("_id", ""))))
        return [self._source(hit, "hybrid_rrf", score) for hit, score in ordered[: context.top_k]]
