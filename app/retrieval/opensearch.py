import asyncio
from typing import Any, Protocol

from app.agent.context import RuntimeContext


class SearchClient(Protocol):
    def search(
        self, *, index: str, body: dict[str, object], params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        ...


class OpenSearchRetriever:
    """BM25 or real OpenSearch hybrid lexical+neural retrieval."""

    def __init__(
        self,
        client: SearchClient,
        index: str = "rag-chunks",
        neural_model_id: str | None = None,
        vector_field: str = "embedding",
        search_pipeline: str | None = None,
    ) -> None:
        self.client = client
        self.index = index
        self.neural_model_id = neural_model_id
        self.vector_field = vector_field
        self.search_pipeline = search_pipeline

    def _filter(self, context: RuntimeContext) -> dict[str, object]:
        filters: list[dict[str, object]] = [
            {"term": {"tenant_id.keyword": context.tenant_id}},
            {"term": {"project_id.keyword": context.project_id}},
        ]
        if context.categories:
            filters.append({"terms": {"category.keyword": list(context.categories)}})
        return {"bool": {"filter": filters}}

    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        category_filter = self._filter(context)
        params: dict[str, str] = {}
        if context.use_hybrid:
            if not self.neural_model_id:
                raise RuntimeError("Hybrid search requires OPENSEARCH_NEURAL_MODEL_ID")
            lexical: dict[str, object] = {"match": {"text": query}}
            neural_body: dict[str, object] = {
                "query_text": query,
                "model_id": self.neural_model_id,
                "k": context.top_k,
            }
            if category_filter:
                neural_body["filter"] = category_filter
                lexical = {"bool": {"must": [lexical], "filter": [category_filter]}}
            body: dict[str, object] = {
                "size": context.top_k,
                "query": {
                    "hybrid": {
                        "queries": [
                            lexical,
                            {"neural": {self.vector_field: neural_body}},
                        ]
                    }
                },
            }
            if self.search_pipeline:
                params["search_pipeline"] = self.search_pipeline
        else:
            bool_query: dict[str, object] = {"must": [{"match": {"text": query}}]}
            if category_filter:
                bool_query["filter"] = [category_filter]
            body = {"size": context.top_k, "query": {"bool": bool_query}}

        response = await asyncio.to_thread(
            self.client.search,
            index=self.index,
            body=body,
            params=params or None,
        )
        results: list[dict[str, object]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source", {}))
            source["score"] = float(hit.get("_score") or 0.0)
            source["search_mode"] = "hybrid" if context.use_hybrid else "bm25"
            results.append(source)
        return results
