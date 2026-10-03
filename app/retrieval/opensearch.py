from opensearchpy import AsyncOpenSearch
from app.agent.context import RuntimeContext


class OpenSearchRetriever:
    """OpenSearch adapter with request-level search mode and category filtering."""

    def __init__(self, client: AsyncOpenSearch, index: str = "rag-chunks") -> None:
        self.client = client
        self.index = index

    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        filters: list[dict[str, object]] = []
        if context.categories:
            filters.append({"terms": {"category.keyword": list(context.categories)}})
        # Hybrid deployments can replace this lexical clause with a search pipeline
        # combining BM25 and a vector/neural query. The runtime flag remains explicit.
        body: dict[str, object] = {
            "size": context.top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": filters,
                }
            },
        }
        response = await self.client.search(index=self.index, body=body)
        results: list[dict[str, object]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source", {}))
            source["score"] = float(hit.get("_score") or 0.0)
            source["search_mode"] = "hybrid" if context.use_hybrid else "bm25"
            results.append(source)
        return results
